import sys
sys.path.append("..")

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models

from matplotlib.ticker import PercentFormatter

from scipy.ndimage import zoom

from NNUtils import CustomDataset
import glob
import tifffile as tiff

import errno
import imageio

from PIL import Image
from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn

from Loss import dice_loss


class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()


def activation_plots(model, image, label, model_name, loss_tag, save_location, image_name):

    b, c, h, w = image.size()

    for name, layer in model.named_modules():

        if isinstance(layer, nn.ReLU):

            model.zero_grad()

            hook_forward = Hook(layer)
            hook_backward = Hook(layer, backward=True)

            if model_name == 'unet':
                output = model(image)
                output = torch.sigmoid(output)
                if loss_tag == 'dice':
                    loss = dice_loss(output, label)
                    loss.backward()

            cam = hook_forward.output
            cam_plot = cam.sum(1, keepdim=True).cpu().data.numpy().squeeze()
            hh, ww = np.shape(cam_plot)
            ratio = h // hh
            cam_plot = zoom(cam_plot, ratio, order=1)

            gradient = hook_backward.output
            gradient = gradient[0]
            gradient = F.relu(gradient)
            gradient_plot = gradient.sum(1, keepdim=True).cpu().data.numpy().squeeze()
            gradient_plot = zoom(gradient_plot, ratio, order=1)

            cam_weighted = cam*gradient
            cam_weighted_plot = cam_weighted.sum(1, keepdim=True).cpu().data.numpy().squeeze()
            cam_weighted_plot = zoom(cam_weighted_plot, ratio, order=1)

            # plots:
            save_folder_model = save_location + '/' + model_name
            try:
                os.mkdir(save_folder_model)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

            save_folder_imgs = save_folder_model + '/' + image_name
            try:
                os.mkdir(save_folder_imgs)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

            img_plot_savename = save_folder_imgs + '/' + 'image.png'
            label_plot_savename = save_folder_imgs + '/' + 'label.png'

            cam_plot_savename = save_folder_imgs + '/' + name + '_cam.png'
            gradient_plot_savename = save_folder_imgs + '/' + name + '_gradient.png'
            weighted_plot_savename = save_folder_imgs + '/' + name + '_weighted_cam.png'

            hist_cam_plot_savename = save_folder_imgs + '/' + name + '_cam_hist.png'
            hist_gradient_plot_savename = save_folder_imgs + '/' + name + '_gradient_hist.png'
            hist_weighted_plot_savename = save_folder_imgs + '/' + name + '_weighted_cam_hist.png'

            norm_hist_cam_plot_savename = save_folder_imgs + '/' + name + '_cam_hist_norm.png'
            norm_hist_gradient_plot_savename = save_folder_imgs + '/' + name + '_gradient_hist_norm.png'
            norm_hist_weighted_plot_savename = save_folder_imgs + '/' + name + '_weighted_cam_hist_norm.png'

            plt.imsave(img_plot_savename, image.cpu().data.numpy().squeeze(), cmap="gray")
            plt.imsave(label_plot_savename, label.cpu().data.numpy().squeeze(), cmap="gray")

            plt.imsave(cam_plot_savename, cam_plot, cmap="jet")
            plt.imsave(gradient_plot_savename, gradient_plot, cmap="jet")
            plt.imsave(weighted_plot_savename, cam_weighted_plot, cmap="jet")

            # histogram plots:
            cam_hist = cam_plot[cam_plot > 0.0]
            gradient_hist = gradient_plot[gradient_plot > 0.0]
            cam_weighted_hist = cam_weighted_plot[cam_weighted_plot > 0.0]

            cam_hist_norm = (cam_hist - np.min(cam_hist)) / (np.max(cam_hist) - np.min(cam_hist))
            gradient_hist_norm = (gradient_hist - np.min(gradient_hist)) / (np.max(gradient_hist) - np.min(gradient_hist))
            cam_weighted_hist_norm = (cam_weighted_hist - np.min(cam_weighted_hist)) / (np.max(cam_weighted_hist) - np.min(cam_weighted_hist))

            # f0 = plt.hist(cam_hist.flatten(),
            #               bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            #               density=True,
            #               color='b',
            #               edgecolor='k',
            #               alpha=0.8)
            # plt.gca().yaxis.set_major_formatter(PercentFormatter())
            # plt.savefig(hist_cam_plot_savename)
            # plt.close()
            #
            # f1 = plt.hist(gradient_hist.flatten(),
            #               bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            #               density=True,
            #               color='b',
            #               edgecolor='k',
            #               alpha=0.8)
            # plt.gca().yaxis.set_major_formatter(PercentFormatter())
            # plt.gcf().savefig(hist_gradient_plot_savename)
            # plt.close()
            #
            # f2 = plt.hist(cam_weighted_hist.flatten(),
            #               bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            #               density=True,
            #               color='b',
            #               edgecolor='k',
            #               alpha=0.8)
            # plt.gca().yaxis.set_major_formatter(PercentFormatter())
            # plt.savefig(hist_weighted_plot_savename)
            # plt.close()

            # normalise
            f3 = plt.hist(cam_hist_norm.flatten(),
                          bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                          density=True,
                          color='b',
                          edgecolor='k',
                          alpha=0.8)
            plt.gca().yaxis.set_major_formatter(PercentFormatter())
            plt.savefig(norm_hist_cam_plot_savename)
            plt.close()

            f4 = plt.hist(gradient_hist_norm.flatten(),
                          bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                          density=True,
                          color='b',
                          edgecolor='k',
                          alpha=0.8)
            plt.gca().yaxis.set_major_formatter(PercentFormatter())
            plt.savefig(norm_hist_gradient_plot_savename)
            plt.close()

            f5 = plt.hist(cam_weighted_hist_norm.flatten(),
                         bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                         density=True,
                         color='b',
                         edgecolor='k',
                         alpha=0.8)
            plt.gca().yaxis.set_major_formatter(PercentFormatter())
            plt.savefig(norm_hist_weighted_plot_savename)
            plt.close()

        print(name + ' plots done of image' + image_name)


def activation_plots_all(data_loc, model, pic_no, save_folder, modelname='unet', loss='dice'):

    device = torch.device('cuda')
    test_data = data_loc
    test_data_imgs = test_data + '/patches'
    test_data_lbls = test_data + '/labels'
    dataset = CustomDataset(test_data_imgs, test_data_lbls, 'none', 3)

    model = torch.load(model)
    model.to(device=device)
    model.eval()

    for i in range(0, pic_no-1, 1):
        testimg_original, labels_original, imagename = dataset[i]
        testimg_original = testimg_original[0, :, :]
        testimg = torch.from_numpy(testimg_original).float().unsqueeze_(0).unsqueeze_(0).to(device=device)
        labels = torch.from_numpy(labels_original).float().unsqueeze_(0).unsqueeze_(0).to(device=device)
        activation_plots(model=model,
                         image=testimg,
                         label=labels,
                         model_name=modelname,
                         loss_tag=loss,
                         save_location=save_folder,
                         image_name=imagename)


if __name__ == '__main__':

    model_name_ori = 'Unet_labelled_repeat_1_augmentation_all_lr_2e-05_epoch_50_CARVE2014_7_r176_s50_Final'
    model_path = '/home/moucheng/PhD/MICCAI 2021/models/7s50'

    test_data = '/home/moucheng/projects_data/Pulmonary_data/CARVE2014/cross_validation/sparse_7_r176_s50/test'
    save_path = '/home/moucheng/Desktop/IPMI_results/30/preliminary/baselines/visual_results'
    save_folder = '/home/moucheng/PhD/BMVC 2021/analysis'

    model = model_name_ori + '.pt'
    model = model_path + '/' + model

    activation_plots_all(data_loc=test_data,
                         model=model,
                         pic_no=100,
                         save_folder=save_folder,
                         )

print('End')









