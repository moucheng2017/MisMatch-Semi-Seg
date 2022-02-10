import os
import errno
import torch
import timeit

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.functional as F
from torch.utils import data

# import Image
# from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances
from scipy import spatial
from sklearn.metrics import mean_squared_error
from torch.optim import lr_scheduler
from NNLoss import dice_loss
from NNMetrics import segmentation_scores, f1_score, hd95
from NNUtils import CustomDataset
from tensorboardX import SummaryWriter
from torch.autograd import Variable

# ===================================================

from ERFNet import ERFAnet
from Baselines import UNet
from Baselines import MTASSUnet


def getData(data_directory, dataset_name, dataset_tag, train_batchsize, data_augment):
    #
    train_image_folder = data_directory + dataset_name + '/' + \
        dataset_tag + '/train/patches'
    train_label_folder = data_directory + dataset_name + '/' + \
        dataset_tag + '/train/labels'
    validate_image_folder = data_directory + dataset_name + '/' + \
        dataset_tag + '/validate/patches'
    validate_label_folder = data_directory + dataset_name + '/' + \
        dataset_tag + '/validate/labels'
    test_image_folder = data_directory + dataset_name + '/' + \
        dataset_tag + '/test/patches'
    test_label_folder = data_directory + dataset_name + '/' + \
        dataset_tag + '/test/labels'
    #
    train_dataset = CustomDataset(train_image_folder, train_label_folder, 'none', 3)
    #
    validate_dataset = CustomDataset(validate_image_folder, validate_label_folder, 'none', 3)
    #
    test_dataset = CustomDataset(test_image_folder, test_label_folder, 'none', 3)
    #
    trainloader = data.DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=1, drop_last=False)
    #
    validateloader = data.DataLoader(validate_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    #
    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    #
    return trainloader, validateloader, testloader, train_dataset, validate_dataset, test_dataset


def plot_segmentation(model_path, data_path, save_path):

    model = torch.load(model_path)

    test_image_folder = data_path + '/patches'

    test_label_folder = data_path + '/labels'

    test_dataset = CustomDataset(test_image_folder, test_label_folder, 'none', 3)

    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    try:

        os.mkdir(save_path)

    except OSError as exc:

        if exc.errno != errno.EEXIST:

            raise

        pass

    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device=device)

    with torch.no_grad():

        # for ii, (test_img, test_label, test_name) in enumerate(testloader):
        for ii, (test_images1, test_images2, test_images3, test_label, test_imagename) in enumerate(testloader):

            test_img = test_images2.to(device=device, dtype=torch.float32)
            # test_img = test_img.to(device=device, dtype=torch.float32)
            test_label = test_label.to(device=device, dtype=torch.float32)

            assert torch.max(test_label) != 100.0

            if 'ERF' in model_path:

                test_outputs_fp, test_outputs_fn = model(test_img)
                test_outputs_fp = torch.sigmoid(test_outputs_fp)
                test_outputs_fn = torch.sigmoid(test_outputs_fn)

                test_class_outputs = (test_outputs_fn + test_outputs_fp) / 2

            elif 'MTASSUnet' in model_path:

                test_outputs, test_attention_outputs = model(test_img)
                test_class_outputs = torch.sigmoid(test_outputs)

            elif 'Entropy_minimisation' in model_path:

                test_outputs = model(test_img)
                test_class_outputs = torch.sigmoid(test_outputs)

            else:
                test_outputs = model(test_img)
                test_class_outputs = torch.sigmoid(test_outputs)

            # save segmentation:
            save_name = save_path + '/test_' + str(ii) + '_seg.png'
            save_name_label = save_path + '/test_' + str(ii) + '_label.png'
            save_name_input = save_path + '/test_' + str(ii) + '_input.png'
            (b, c, h, w) = test_label.shape
            assert c == 1
            test_class_outputs = test_class_outputs.reshape(h, w).cpu().detach().numpy() > 0.5
            plt.imsave(save_name, test_class_outputs, cmap='gray')
            plt.imsave(save_name_label, test_label.reshape(h, w).cpu().detach().numpy(), cmap='gray')
            plt.imsave(save_name_input, test_img.reshape(h, w).cpu().detach().numpy(), cmap='gray')

            # save_name_overlapped = save_path + '/test_' + str(ii) + '_overlapped.png'
            # test_img = test_img.reshape(h, w).cpu().detach().numpy().reshape(h, w, 1)
            # new_test_img = np.concatenate((test_img, test_img, test_img), axis=2).astype(np.float32)
            # print(np.unique(new_test_img))
            #
            # new_seg = np.zeros((h, w, 1))
            # test_class_outputs = test_class_outputs.reshape(h, w, 1)
            # new_seg = np.concatenate((test_class_outputs, new_seg, new_seg), axis=2).astype(np.float32)
            # print(np.unique(new_seg))
            # overlapped = new_seg + new_test_img
            #
            # plt.imshow(overlapped)
            # plt.show()
            # plt.imsave(save_name_overlapped, overlapped)


if __name__ == '__main__':

    model_path = '/home/moucheng/Desktop/IPMI_results/30/preliminary/baselines/Unet_labelled_repeat_1_augmentation_all_lr_0.0002_epoch_50_CARVE2014_3d_c0_r176_s30_Final.pt'
    data_path = '/home/moucheng/projects_data/lung/public_data_sets/CARVE2014/cross_validation/3d_c0_r176_s30/test'
    save_path = '/home/moucheng/Desktop/IPMI_results/30/preliminary/baselines/visual_results'
    plot_segmentation(model_path, data_path, save_path)

print('End')