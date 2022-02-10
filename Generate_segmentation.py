import torch
import random
import numpy as np
import os

import timeit
import glob
import tifffile as tiff

import errno
import imageio

from PIL import Image
from torch.utils import data

from NNUtils import CustomDataset, CustomDataset_MT, CustomDataset_FixMatch


# ====================================================================================
# ====================================================================================
def plot(model, modelname, save_location, data, dim, test_img_no):
    #
    device = torch.device('cuda')
    #
    model = model.to(device=device)
    #
    model.eval()

    inferencetime = 0.0

    with torch.no_grad():
        #
        # evaluate_index_all = range(0, len(data) - 1)
        evaluate_index_all = range(0, test_img_no-1)
        # evaluate_index_all = range(0, 10)
        #
        for index in evaluate_index_all:
            # extract a few random indexs every time in a range of the data
            if dim == 3:
                testimg1, testimg2, testimg3, testlabel, test_imagename = data[index]
                #
                # augmentation = random.random()
                # #
                # if augmentation > 0.5:
                #     c, h, w = np.shape(testimg)
                #     for channel in range(c):
                #         testimg[channel, :, :] = np.flip(testimg[channel, :, :], axis=0).copy()
                #
                # ========================================================================
                # ========================================================================
                testimg1 = torch.from_numpy(testimg1).to(device=device, dtype=torch.float32)
                testimg2 = torch.from_numpy(testimg2).to(device=device, dtype=torch.float32)
                testimg3 = torch.from_numpy(testimg3).to(device=device, dtype=torch.float32)

                testlabel = torch.from_numpy(testlabel).to(device=device, dtype=torch.float32)

                c, h, w = testimg2.size()
                testimg1 = testimg1.expand(1, c, h, w)
                testimg = testimg2.expand(1, c, h, w)
                testimg3 = testimg3.expand(1, c, h, w)

            # inference starts:
            torch.cuda.synchronize(device)
            start = timeit.default_timer()

            if 'Ensemble' in modelname:
                #
                testoutput_original_fn, testoutput_original_fp = model(testimg)
                testoutput_fn = torch.sigmoid(testoutput_original_fn.view(1, h, w))
                testoutput_fp = torch.sigmoid(testoutput_original_fp.view(1, h, w))
                #
                testinverse_prob_outputs_fn = torch.ones_like(testoutput_fn).to(device=device, dtype=torch.float32)
                testinverse_prob_outputs_fn = testinverse_prob_outputs_fn - testoutput_fn
                testoutput = (testoutput_fp + testinverse_prob_outputs_fn) / 2
                #
            elif 'AttentionUNet' in modelname:
                #
                testoutput_original, attention_weights, trunk_features = model(testimg)
                #
                testoutput = torch.sigmoid(testoutput_original.view(1, h, w))
                #
            elif 'FalsePositiveUNet' in modelname or 'FalseNegativeUNet' in modelname or 'Negative' in model_name:

                testoutput_original, attention_weights, attention_weights_, trunk_features = model(testimg)
                testoutput = torch.sigmoid(testoutput_original.view(1, h, w))

            elif 'ERF' in modelname:
                test_outputs_fp, test_outputs_fn, test_pseudo_x1_a, test_pseudo_x1_b, test_pseudo_x2, test_pseudo_x3_a, test_pseudo_x3_b, test_x_, test_x__, test_x___ = model(testimg1, testimg, testimg3)
                test_outputs_fp = torch.sigmoid(test_outputs_fp)
                test_outputs_fn = torch.sigmoid(test_outputs_fn)
                testoutput = (test_outputs_fn + test_outputs_fp) / 2

            elif 'Morph' in modelname:

                val_output1, val_output2 = model(testimg)
                testoutput = (val_output1 + val_output2) / 2

            elif 'CCT' in modelname:
                testoutput, _, __ = model(testimg)

            elif 'MTA' in modelname:
                testoutput, _ = model(testimg)

            else:
                #
                testoutput_original = model(testimg)
                #
                # print(testoutput_original)
                #
                testoutput = torch.sigmoid(testoutput_original.view(1, h, w))
                #
                # testoutput = torch.sigmoid(testoutput_original)
                #

            torch.cuda.synchronize(device)
            stop = timeit.default_timer()

            inferencetime += stop - start
            # print('Inference Time: ', stop - start)

            threshold = torch.tensor([0.5], dtype=torch.float32, device=device, requires_grad=False)

            upper = torch.tensor([1.0], dtype=torch.float32, device=device, requires_grad=False)

            lower = torch.tensor([0.0], dtype=torch.float32, device=device, requires_grad=False)

            testoutput = torch.where(testoutput > threshold, upper, lower)

            try:
                os.mkdir(save_location)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass
            # Plot error maps:
            #
            testimg = testimg2.cpu().squeeze().detach().numpy()
            # testimg = testimg[2, :, :]
            testimg = np.asarray(testimg, dtype=np.float32)
            #
            label = testlabel.squeeze().cpu().detach().numpy()
            pred = testoutput.squeeze().cpu().detach().numpy()
            #
            label = np.asarray(label, dtype=np.uint8)
            pred = np.asarray(pred, dtype=np.uint8)
            #
            # if augmentation > 0.5:
            #     pred = np.flip(pred, axis=0).copy()
            #     testimg = np.flip(testimg, axis=0).copy()
            #
            # difference = label - pred
            # addition = label + pred
            #
            # error_map = np.zeros((h, w, 3), dtype=np.uint8)
            # #
            # error_map[difference == -1] = [255, 0, 0]  # false positive red
            # error_map[difference == 1] = [0, 0, 255]  # false negative blue
            # # error_map[addition == 0] = [0, 255, 0]  # true negative green
            # error_map[addition == 2] = [255, 255, 0]  # true positive yellow

            error_map = np.zeros((h, w, 3), dtype=np.uint8)
            gt_map = np.zeros((h, w, 3), dtype=np.uint8)

            for hh in range(0, h):
                for ww in range(0, w):
                    label_ = label[hh, ww]
                    pred_ = pred[hh, ww]
                    pixel = testimg[hh, ww]
                    if label_ == 1 and pred_ == 1:
                        error_map[hh, ww, 0] = 0
                        error_map[hh, ww, 1] = 255
                        error_map[hh, ww, 2] = 0
                    # elif label_ == 0 and pred_ == 0:
                    #     error_map[hh, ww, 0] = 0
                    #     error_map[hh, ww, 1] = 255
                    #     error_map[hh, ww, 2] = 0
                    elif label_ == 0 and pred_ == 1:
                        error_map[hh, ww, 0] = 255
                        error_map[hh, ww, 1] = 0
                        error_map[hh, ww, 2] = 0
                    elif label_ == 1 and pred_ == 0:
                        error_map[hh, ww, 0] = 0
                        error_map[hh, ww, 1] = 0
                        error_map[hh, ww, 2] = 255

                    # if pixel < -1:
                    #     error_map[hh, ww, :] = 0

                    if label_ == 1:
                        gt_map[hh, ww, 0] = 255
                        gt_map[hh, ww, 1] = 255
                        gt_map[hh, ww, 2] = 0

            prediction_name = 'seg_' + test_imagename + '.png'
            full_error_map_name = os.path.join(save_location, prediction_name)
            imageio.imsave(full_error_map_name, error_map)
            #
            pic_name = 'original_' + test_imagename + '.png'
            full_pic_map_name = os.path.join(save_location, pic_name)

            gt_name = 'gt_' + test_imagename + '.png'
            full_gt_map_name = os.path.join(save_location, gt_name)
            imageio.imsave(full_gt_map_name, gt_map)
            # check testimg shape:
            # c, h, w = testimg.shape
            # if c == 3:
            #     testimg = np.reshape(testimg, (h, w, c))
            # imageio.imsave(full_pic_map_name, testimg)
            # #
            # seg_img = Image.open(full_error_map_name)
            # input_img = Image.open(full_pic_map_name)
            # #
            # seg_img = seg_img.convert("RGBA")
            # input_img = input_img.convert("RGBA")
            # #
            # alphaBlended = Image.blend(seg_img, input_img, alpha=.6)
            # blended_name = 'blend_' + test_imagename + '.png'
            # full_blend_map_name = os.path.join(save_location, blended_name)
            # imageio.imsave(full_blend_map_name, alphaBlended)
            #
            # os.remove(full_pic_map_name)
            # os.remove(full_error_map_name)
            #
            imageio.imsave(full_pic_map_name, testimg)

            seg_img = Image.open(full_error_map_name)
            input_img = Image.open(full_pic_map_name)

            seg_img = seg_img.convert("RGBA")
            input_img = input_img.convert("RGBA")

            alphaBlended = Image.blend(seg_img, input_img, alpha=.6)
            blended_name = 'blend_' + test_imagename + '.png'
            full_blend_map_name = os.path.join(save_location, blended_name)
            imageio.imsave(full_blend_map_name, alphaBlended)

            gt_img = Image.open(full_gt_map_name)
            input_img = Image.open(full_pic_map_name)

            gt_img = gt_img.convert("RGBA")
            input_img = input_img.convert("RGBA")

            betaBlended = Image.blend(gt_img, input_img, alpha=.6)
            imageio.imsave(full_gt_map_name, betaBlended)

            if 'ERF' not in modelname:
                os.remove(full_pic_map_name)
                os.remove(full_error_map_name)
                os.remove(full_gt_map_name)
            else:
                pass

            print(full_error_map_name + ' is saved.')

    return inferencetime/(len(evaluate_index_all)+1)


if __name__ == '__main__':

    fold = 7

    name1 = 'ERFAnetZ_repeat_1_augmentation_none_alpha_0.002_lr_2e-05_epoch_50annealing_down_at_0_beta_0.8_constraint_jacobi_all_gamma_0.2_CARVE2014_' + str(fold) + '_r176_s50_Final.pt'
    name2 = 'FixMatch_repeat_1_alpha_0.002_lr_2e-05_epoch_50_annealing_down_at_0_beta_0.8_CARVE2014_' + str(fold) + '_r176_s50_Final.pt'
    name3 = 'MeanTeacher_repeat_1_alpha_0.002_lr_2e-05_epoch_50_annealing_down_at_0_beta_0.8_CARVE2014_' + str(fold) + '_r176_s50_Final.pt'
    name4 = 'Unet_labelled_repeat_1_augmentation_all_lr_2e-05_epoch_50_CARVE2014_' + str(fold) + '_r176_s50_Final.pt'
    # name5 = 'Old_FalseNegativeUNet_Trainbatch_40_Valbatch_4_width_32_repeat_3_loss_dice_augment_True_lr_decay_True_attention_no_3_down8_down8_down8_exp4_depthwise_vehicle_e60_lr0.0002_Final.pt'

    # name1 = 'CCT_repeat_1_alpha_0.002_lr_2e-05_epoch_40_annealing_down_at_0_beta_0.5_CARVE2014_sparse_3_r176_s50_Final.pt'
    # name2 = 'Morph_repeat_1_alpha_0.002_lr_2e-05_epoch_40_annealing_down_at_0_beta_0.5_CARVE2014_sparse_3_r176_s50_Final.pt'
    # name3 = 'MTASSUnet_repeat_1_augmentation_True_alpha_0.002_lr_2e-05_epoch_50_CARVE2014_3_r176_s50_Final.pt'

    model_names = [name1, name2, name3, name4]

    for model_name in model_names:
        # change here:

        mother_path = '/home/moucheng/PhD/MICCAI 2021/models/' + str(fold) + 's50/'
        data_folder = '/home/moucheng/projects_data/Pulmonary_data/CARVE2014/cross_validation/sparse_' + str(fold) + '_r176_s30'

        print(model_name)
        path = mother_path + model_name
        print(path)
        # model = torch.model(path)
        model = torch.load(path)
        print('Model is loaded.')
        # ============================
        # There are issues with models trained with different versions of libraries.
        # models trained on cluster can't do inferences locally... different versions in libraries
        # To solve this:
        # 1. call a new empty model of testing model
        # 2. load the weights in the new emtpy model with weights from testing model
        # 3. load the new model, this avoids the differences of versions of libraies
        # ============================
        # model_dict = model.state_dict()
        # print(model_dict)
        # new_model = AttentionUNet(in_ch=3, width=16, visulisation=True)
        # new_model = FPAttentionUNet(in_ch=3, width=16, fpa_type='2', dilation=6, attention_no=3, width_expansion=1, param_share_times=2, visulisation=True)
        # new_model = FNAttentionUNet(in_ch=3, width=16, attention_no=3, width_expansion=4, attention_type='2', visulisation=True)
        # new_model_dict = new_model.state_dict()
        # model_dict = {k: v for k, v in model_dict.items() if k in new_model_dict.keys()}
        # new_model_dict.update(model_dict)
        # new_path = '/home/moucheng/Results/cityscape/models/new_' + model_name
        # torch.save(new_model, new_path)
        # #
        # # model = new_model.load_state_dict(torch.load(new_path))
        # model = torch.load(new_path)
        # =====================================================================

        # validate_image_folder = data_folder + str(fold_no) + '/validate/patches'
        # validate_label_folder = data_folder + str(fold_no) + '/validate/labels'
        validate_image_folder = data_folder + '/test/patches'
        validate_label_folder = data_folder + '/test/labels'
        data = CustomDataset(validate_image_folder, validate_label_folder, False, dimension=3)

        save_location = mother_path + 'segmentation'

        try:
            os.mkdir(save_location)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        save_location = save_location + '/' + model_name[: -3]

        inferencetime = plot(model=model, save_location=save_location, data=data, modelname=model_name, dim=3, test_img_no=10)

        print("inference time:" + str(inferencetime))

    # fold_no = 2
    # for model_name in model_names:
    #     path = '/home/moucheng/projects_codes/saved_models/' + model_name
    #     model = torch.load(path)
    #     print('Model is loaded.')
    #     data_folder = '/home/moucheng/projects_data/Brain_data/BRATS2018/MICCAI_BraTS_2018_Data_Training/ET_L0_H210_f5/'
    #     validate_image_folder = data_folder + str(fold_no) + '/validate/patches'
    #     validate_label_folder = data_folder + str(fold_no) + '/validate/labels'
    #     data = CustomDataset(validate_image_folder, validate_label_folder, False)
    #     #
    #     save_location = '/home/moucheng/projects_data/Segmentation/fold_' + str(fold_no)
    #     #
    #     try:
    #         os.mkdir(save_location)
    #     except OSError as exc:
    #         if exc.errno != errno.EEXIST:
    #             raise
    #         pass
    #     #
    #     save_location = save_location + '/' + model_name[: -3]
    #     #
    #     plot(model=model, save_location=save_location, data=data, modelname=model_name)
    # #
    # fold_no = 4
    # for model_name in model_names:
    #     path = '/home/moucheng/projects_codes/saved_models/' + model_name
    #     model = torch.load(path)
    #     print('Model is loaded.')
    #     data_folder = '/home/moucheng/projects_data/Brain_data/BRATS2018/MICCAI_BraTS_2018_Data_Training/ET_L0_H210_f5/'
    #     validate_image_folder = data_folder + str(fold_no) + '/validate/patches'
    #     validate_label_folder = data_folder + str(fold_no) + '/validate/labels'
    #     data = CustomDataset(validate_image_folder, validate_label_folder, False)
    #     #
    #     save_location = '/home/moucheng/projects_data/Segmentation/fold_' + str(fold_no)
    #     #
    #     try:
    #         os.mkdir(save_location)
    #     except OSError as exc:
    #         if exc.errno != errno.EEXIST:
    #             raise
    #         pass
    #     #
    #     save_location = save_location + '/' + model_name[: -3]
    #     #
    #     plot(model=model, save_location=save_location, data=data, modelname=model_name)
    #     #
print('End')
