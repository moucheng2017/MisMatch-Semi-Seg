import os
import csv
import numpy as np
import pathlib
import scipy
import math
import sys
import argparse
import nibabel as nib
import torch
from pathlib import Path
sys.path.append('')
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.transform import resize
from test_util import calculate_metric_percase
from networks.vnet import VNet
from networks.vnet_mismatch import VNetMisMatch
from tqdm import tqdm

# This prints out a grid: rows are different images, each row has three columns of image, label, and overlapped image

parser = argparse.ArgumentParser('Run inference on BRATS')
parser.add_argument('--img_source', type=str, help='source image file', default='/home/moucheng/projects_data/Task01_BrainTumour/test/imgs')
parser.add_argument('--lbl_source', type=str, help='source label file', default='/home/moucheng/projects_data/Task01_BrainTumour/test/lbls')

# parser.add_argument('--model_source', type=str, help='model path', default='/home/moucheng/projects/2023_03_14_TMI/new_results_3d_lung_brain/model/Task01_BrainTumour/UAT_brain_exp1_c1.0/iter_2000.pth')
# parser.add_argument('--model_name', type=str, default='ua_mt', help='model name, use mismatch for ours and ua_mt for baseline')

parser.add_argument('--model_source', type=str, help='model path', default='/home/moucheng/projects/2023_03_14_TMI/new_results_3d_lung_brain/model_mismatch/Task01_BrainTumour/MisMatch_brain_exp11_c1.0_d_True_di_9/iter_1300.pth')
parser.add_argument('--model_name', type=str, default='mismatch', help='model name, use mismatch for ours and ua_mt for baseline')

parser.add_argument('--new_dim_d', type=int, help='new dimension d', default=128)
parser.add_argument('--new_dim_h', type=int, help='new dimension h', default=128)
parser.add_argument('--new_dim_w', type=int, help='new dimension w', default=96)
parser.add_argument('--test_cases', type=int, help='number of testing cases', default=10)
args = parser.parse_args()


if __name__ == '__main__':

    all_cases = [os.path.join(args.img_source, f) for f in os.listdir(args.img_source)]
    all_cases.sort()
    all_cases = all_cases[:args.test_cases]

    all_labels = [os.path.join(args.lbl_source, f) for f in os.listdir(args.lbl_source)]
    all_labels.sort()
    all_labels = all_labels[:args.test_cases]

    segmentation_iou_all_cases_dice = []
    segmentation_iou_all_cases_jc = []
    segmentation_iou_all_cases_hd = []
    segmentation_iou_all_cases_asd = []

    if args.model_name == 'ua_mt':
        # print(torch.load(args.model_source))
        net = VNet(n_channels=4, n_classes=2, n_filters=8,  normalization='instancenorm', has_dropout=False).cuda()
        # save_mode_path = args.model_source
        net.load_state_dict(torch.load(args.model_source))
    elif args.model_name == 'mismatch':
        net = VNetMisMatch(n_channels=4, n_classes=2, n_filters=8, normalization='instancenorm', has_dropout=False, dilation=6).cuda()
        save_mode_path = args.model_source
        net.load_state_dict(torch.load(args.model_source))
    else:
        raise NotImplementedError

    for case_index, (each_case, each_label) in tqdm(enumerate(zip(all_cases, all_labels))):

        img_volume = nib.load(each_case)
        img_volume = img_volume.get_fdata()
        img = np.asfarray(img_volume)

        lbl_volume = nib.load(each_label)
        lbl_volume = lbl_volume.get_fdata()
        lbl = np.asfarray(lbl_volume).astype(int)

        # if args.transpose > 0:
        if len(np.shape(img)) == 3:
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            lbl = np.transpose(lbl, (2, 0, 1))
        elif len(np.shape(img)) == 4:
            img = np.transpose(img, (3, 2, 0, 1))
            lbl = np.transpose(lbl, (2, 0, 1))
        else:
            raise NotImplementedError

        lbl[lbl > 0] = 1

        # Path(args.save_path).mkdir(parents=True, exist_ok=True)

        # if case_index == 0:
        #     print(np.shape(img))

        division_d = np.shape(img)[-3] // args.new_dim_d
        division_h = np.shape(img)[-2] // args.new_dim_h
        division_w = np.shape(img)[-1] // args.new_dim_w

        d_start = (np.shape(img)[-3] - division_d*args.new_dim_d) // 2
        d_end = d_start + division_d*args.new_dim_d

        h_start = (np.shape(img)[-2] - division_h*args.new_dim_h) // 2
        h_end = h_start + division_h*args.new_dim_h

        w_start = (np.shape(img)[-1] - division_w*args.new_dim_w) // 2
        w_end = w_start + division_w*args.new_dim_w

        img = img[:, d_start:d_end, h_start:h_end, w_start:w_end]
        lbl = lbl[d_start:d_end, h_start:h_end, w_start:w_end]

        # if case_index == 0:
        #     print(np.shape(img))

        seg = np.zeros((np.shape(img)[-3], np.shape(img)[-2], np.shape(img)[-1]))

        imgs_d = np.split(img, np.shape(img)[-3] // args.new_dim_d, axis=1)

        count = 0

        for i, each_img_d in enumerate(imgs_d):
            imgs_d_h = np.split(each_img_d, np.shape(img)[-2] // args.new_dim_h, axis=2)
            for j, each_img_h in enumerate(imgs_d_h):
                imgs_d_h_w = np.split(each_img_h, np.shape(img)[-1] // args.new_dim_w, axis=3)
                for k, each_img_w in enumerate(imgs_d_h_w):

                    assert np.shape(each_img_w)[-3] == args.new_dim_d
                    assert np.shape(each_img_w)[-2] == args.new_dim_h
                    assert np.shape(each_img_w)[-1] == args.new_dim_w

                    seg_ = torch.from_numpy(each_img_w).cuda().unsqueeze(0).float()

                    if args.model_name == 'mismatch':
                        y1, y2 = net(seg_)
                        # y = (y1 + y2) / 2
                        y1 = F.softmax(y1, dim=1)
                        y2 = F.softmax(y2, dim=1)
                        seg_ = (y1 + y2) / 2
                        # seg_ = y1
                        # seg_ = F.softmax(y, dim=1)
                    elif args.model_name == 'ua_mt':
                        seg_ = net(seg_)
                        seg_ = F.softmax(seg_, dim=1)
                    else:
                        raise NotImplementedError

                    seg_ = seg_.squeeze().detach().cpu().numpy()
                    seg_ = np.argmax(seg_, axis=0)
                    # print(np.shape(seg_))

                    seg[
                    i*args.new_dim_d:(i+1)*args.new_dim_d,
                    j*args.new_dim_h:(j+1)*args.new_dim_h,
                    k*args.new_dim_w:(k+1)*args.new_dim_w] = seg_

                    count += 1

                    del each_img_w
                    del seg_

        metrics = calculate_metric_percase(seg.squeeze().astype(int), lbl.squeeze().astype(int))
        segmentation_iou_all_cases_dice.append(metrics[0])
        segmentation_iou_all_cases_jc.append(metrics[1])
        segmentation_iou_all_cases_hd.append(metrics[2])
        segmentation_iou_all_cases_asd.append(metrics[3])

    # seg = nib.Nifti1Image(seg, affine=np.eye(4))
    # seg_name = str(args.flag) + '_d' + str(args.new_dim) + '_c' + str(args.confidence) + '_segmentation.nii'
    # save_file = os.path.join(args.save_path, seg_name)
    # nib.save(seg, save_file)

    print('\n')
    print('\n')
    print('Test dice mean is: ' + str(np.nanmean(segmentation_iou_all_cases_dice)))
    print('Test dice std is: ' + str(np.nanstd(segmentation_iou_all_cases_dice)))
    print('\n')
    print('Test jc mean is: ' + str(sum(segmentation_iou_all_cases_jc) / len(segmentation_iou_all_cases_jc)))
    print('Test jc std is: ' + str(np.nanstd(segmentation_iou_all_cases_jc)))
    print('\n')
    print('Test hd mean is: ' + str(sum(segmentation_iou_all_cases_hd) / len(segmentation_iou_all_cases_hd)))
    print('Test hd std is: ' + str(np.nanstd(segmentation_iou_all_cases_hd)))
    print('\n')
    print('Test asd mean is: ' + str(sum(segmentation_iou_all_cases_asd) / len(segmentation_iou_all_cases_asd)))
    print('Test asd std is: ' + str(np.nanstd(segmentation_iou_all_cases_asd)))
    print('Done')




