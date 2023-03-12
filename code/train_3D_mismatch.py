import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet_mismatch import VNetMisMatch
from dataloaders import utils
from utils import ramps, losses
from pathlib import Path
from dataloaders.custom_dataloader import MedicalData3D
from dataloaders.la_heart import ToTensor


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/moucheng/projects_data/Task01_BrainTumour', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='Mismatch_unlabel', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu') # total batch_size including labelled and unlabelled
parser.add_argument('--in_channel', type=int,  default=4, help='number of input channels, 1 for CT, 4 for BRATS')
parser.add_argument('--width', type=int,  default=8, help='number of filters')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
# parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--dilation', type=int,  default=9, help='Dilation rate for positive attention encoder')

parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
parser.add_argument('--detach', default=True,  help='gradient stopping between the consistency regularisation')

parser.add_argument('--workers', type=int,  default=4, help='nnumber of workers')

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model_mismatch/" + str(Path(args.root_path).stem) + '/' + args.exp + '_c' + str(args.consistency) + '_d_' + str(args.detach) + '_di_' + str(args.dilation) + "/"

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (96, 96, 96)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model():
        # Network definition
        net = VNetMisMatch(n_channels=args.in_channel, n_classes=num_classes, n_filters=args.width, normalization='instancenorm', has_dropout=False, dilation=args.dilation)
        model = net.cuda()
        return model

    model = create_model()
    # ema_model = create_model(ema=True)

    db_train_l = MedicalData3D(images_folder=train_data_path + '/labelled/imgs',
                               labels_folder=train_data_path + '/labelled/lbls',
                               output_shape=patch_size,
                               crop_aug=True)

    db_train_u = MedicalData3D(images_folder=train_data_path + '/unlabelled/imgs',
                               labels_folder=None,
                               output_shape=patch_size,
                               crop_aug=True)

    db_test = MedicalData3D(images_folder=train_data_path + '/test/imgs',
                            labels_folder=train_data_path + '/test/lbls',
                            output_shape=patch_size,
                            crop_aug=False)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)

    assert args.batch_size > args.labeled_bs

    trainloader_l = DataLoader(db_train_l, batch_size=args.labeled_bs, num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn, shuffle=True, drop_last=False)
    trainloader_u = DataLoader(db_train_u, batch_size=args.batch_size - args.labeled_bs, num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn, shuffle=True, drop_last=False)
    testloader = DataLoader(db_test, batch_size=1, num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn, shuffle=False)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader_l)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader_l)+1
    lr_ = base_lr

    iter_trainloader_l = iter(trainloader_l)
    iter_trainloader_u = iter(trainloader_u)

    model.train()
    for epoch_num in tqdm(range(max_iterations), ncols=70):

        try:
            img_label_l, _ = next(iter_trainloader_l)
            img_u, _ = next(iter_trainloader_u)
        except StopIteration:
            iter_trainloader_l = iter(trainloader_l)
            iter_trainloader_u = iter(trainloader_u)
            img_label_l, _ = next(iter_trainloader_l)
            img_u, _ = next(iter_trainloader_u)

        volume_batch_l, label_batch_l = img_label_l['image'], img_label_l['label']
        volume_batch_l, label_batch_l = volume_batch_l.cuda(), label_batch_l.cuda()
        volume_batch_u = img_u['image']
        volume_batch_u = volume_batch_u.cuda()
        volume_batch = torch.cat((volume_batch_l, volume_batch_u), dim=0)

        outputs_p, outputs_n = model(volume_batch)

        # calculate the supervised loss:
        loss_seg = 0.5*F.cross_entropy(outputs_p[:labeled_bs], label_batch_l.long()) + 0.5*F.cross_entropy(outputs_n[:labeled_bs], label_batch_l.long())
        outputs_soft_p = F.softmax(outputs_p[labeled_bs:], dim=1)
        outputs_soft_n = F.softmax(outputs_n[labeled_bs:], dim=1)
        outputs_soft_avg = (outputs_soft_p + outputs_soft_n) / 2
        loss_seg_dice = losses.dice_loss(outputs_soft_avg[:labeled_bs, 1, :, :, :], label_batch_l)
        supervised_loss = 0.5*(loss_seg+loss_seg_dice)

        # calculate the unsupervised loss:
        consistency_weight = get_current_consistency_weight(iter_num // 150)
        # mask = F.softmax((outputs_p[labeled_bs:] + outputs_n[labeled_bs:]) / 2, dim=1)
        # mask = (mask > args.threshold) # this might select only half of the digits
        # outputs_p_u = torch.masked_select(outputs_p[labeled_bs:], mask)
        # outputs_n_u = torch.masked_select(outputs_n[labeled_bs:], mask)

        outputs_p_u = outputs_p[labeled_bs:]
        outputs_n_u = outputs_n[labeled_bs:]
        outputs_p_u = F.normalize(outputs_p_u, dim=0, p=1)
        outputs_n_u = F.normalize(outputs_n_u, dim=0, p=1)

        if args.detach is True:
            consistency_dist = 0.5 * torch.nn.MSELoss(reduction='mean')(outputs_p_u, outputs_n_u.detach()) + \
                               0.5 * torch.nn.MSELoss(reduction='mean')(outputs_n_u, outputs_p_u.detach())
        else:
            consistency_dist = 0.5 * torch.nn.MSELoss(reduction='mean')(outputs_p_u, outputs_n_u) + \
                               0.5 * torch.nn.MSELoss(reduction='mean')(outputs_n_u, outputs_p_u)

        consistency_loss = consistency_weight * consistency_dist

        loss = supervised_loss + consistency_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_num = iter_num + 1
        writer.add_scalar('lr', lr_, iter_num)
        writer.add_scalar('loss/loss', loss, iter_num)
        writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
        writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
        writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
        writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
        writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

        logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %
                     (iter_num, loss.item(), consistency_dist.item(), consistency_weight))

        # if iter_num % 50 == 0:
        #     image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
        #     grid_image = make_grid(image, 5, normalize=True)
        #     writer.add_image('train/Image', grid_image, iter_num)
        #
        #     image = torch.max(outputs_soft_avg[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
        #     image = utils.decode_seg_map_sequence(image)
        #     grid_image = make_grid(image, 5, normalize=False)
        #     writer.add_image('train/Predicted_label', grid_image, iter_num)
        #
        #     image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
        #     grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
        #     writer.add_image('train/Groundtruth_label', grid_image, iter_num)
        #
        #     #####
        #     image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
        #     grid_image = make_grid(image, 5, normalize=True)
        #     writer.add_image('unlabel/Image', grid_image, iter_num)
        #
        #     image = torch.max(outputs_soft_avg[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
        #     image = utils.decode_seg_map_sequence(image)
        #     grid_image = make_grid(image, 5, normalize=False)
        #     writer.add_image('unlabel/Predicted_label', grid_image, iter_num)
        #
        #     image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
        #     grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
        #     writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

        ## change lr
        if iter_num % 2500 == 0:
            lr_ = base_lr * 0.1 ** (iter_num // 2500)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

        if iter_num % 1000 == 0:
            save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        save_mode_path = os.path.join(snapshot_path, 'last.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

        if iter_num >= max_iterations:
            break
        # time1 = time.time()
        if iter_num >= max_iterations:
            break

    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
