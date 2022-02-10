import os
import errno
import torch
import timeit

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.functional as F
from torch.utils import data

import glob
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


def trainModels(dataset_tag,
                dataset_name,
                data_directory,
                cross_validation,
                input_dim,
                class_no,
                repeat,
                train_batchsize,
                augmentation,
                num_epochs,
                learning_rate,
                width,
                network,
                log_tag,
                main_loss='dice',
                self_addition=True,
                save_all_segmentation=False):

    for j in range(1, repeat + 1):

        repeat_str = str(j)

        if network == 'ERFAnet':
            #
            Exp = ERFAnet(in_ch=input_dim, width=width, class_no=class_no, identity_add=self_addition)
            #
            Exp_name = 'ERFAnet_labelled' + \
                       '_repeat_' + str(repeat_str) + \
                       '_augmentation_' + str(augmentation) + \
                       '_lr_' + str(learning_rate) + \
                       '_epoch_' + str(num_epochs) + '_' \
                       + dataset_name + '_' + dataset_tag

        elif network == 'Unet':
            #
            Exp = UNet(in_ch=input_dim, width=width, class_no=class_no)
            #
            Exp_name = 'Unet_labelled' + \
                       '_repeat_' + str(repeat_str) + \
                       '_augmentation_' + str(augmentation) + \
                       '_lr_' + str(learning_rate) + \
                       '_epoch_' + str(num_epochs) + '_' \
                       + dataset_name + '_' + dataset_tag

        # ====================================================================================================================================================================
        trainloader, validateloader, testloader, train_dataset, validate_dataset, test_dataset = getData(data_directory, dataset_name, dataset_tag, train_batchsize, augmentation, cross_validation)
        # ===================
        trainSingleModel(Exp,
                         Exp_name,
                         num_epochs,
                         learning_rate,
                         dataset_name,
                         dataset_tag,
                         train_dataset,
                         train_batchsize,
                         trainloader,
                         validateloader,
                         testloader,
                         losstag=main_loss,
                         class_no=class_no,
                         log_tag=log_tag,
                         save_all_segmentation=save_all_segmentation)


def getData(data_directory, dataset_name, dataset_tag, train_batchsize, data_augment, cross_validation):

    if cross_validation is False:

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
        train_dataset = CustomDataset(train_image_folder, train_label_folder, data_augment, 3)
        #
        validate_dataset = CustomDataset(validate_image_folder, validate_label_folder, data_augment, 3)
        #
        test_dataset = CustomDataset(test_image_folder, test_label_folder, 'none', 3)
        #
        trainloader = data.DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=2, drop_last=False)
        #
        validateloader = data.DataLoader(validate_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
        #
        testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    else:

        train_image_folder = data_directory + dataset_name + '/cross_validation/' + \
            dataset_tag + '/train/patches'
        train_label_folder = data_directory + dataset_name + '/cross_validation/' + \
            dataset_tag + '/train/labels'
        validate_image_folder = data_directory + dataset_name + '/cross_validation/' + \
            dataset_tag + '/validate/patches'
        validate_label_folder = data_directory + dataset_name + '/cross_validation/' + \
            dataset_tag + '/validate/labels'
        test_image_folder = data_directory + dataset_name + '/cross_validation/' + \
            dataset_tag + '/test/patches'
        test_label_folder = data_directory + dataset_name + '/cross_validation/' + \
            dataset_tag + '/test/labels'
        #
        train_dataset = CustomDataset(train_image_folder, train_label_folder, data_augment, 3)
        #
        validate_dataset = CustomDataset(validate_image_folder, validate_label_folder, 'none', 3)
        #
        test_dataset = CustomDataset(test_image_folder, test_label_folder, 'none', 3)
        #
        trainloader = data.DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=2, drop_last=False)
        #
        validateloader = data.DataLoader(validate_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
        #
        testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    return trainloader, validateloader, testloader, train_dataset, validate_dataset, test_dataset

# =====================================================================================================================================


def trainSingleModel(model,
                     model_name,
                     num_epochs,
                     learning_rate,
                     dataset_name,
                     dataset_tag,
                     train_dataset,
                     train_batchsize,
                     trainloader,
                     validateloader,
                     testdata,
                     losstag,
                     log_tag,
                     class_no,
                     save_all_segmentation):
    # change log names
    # training_amount = len(train_dataset)
    #
    # iteration_amount = training_amount // train_batchsize - 1
    #
    device = torch.device('cuda')
    #
    save_model_name = model_name
    #
    saved_information_path = '../../Results'
    #
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_information_path = saved_information_path + '/' + dataset_name
    #
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_information_path = saved_information_path + '/' + dataset_tag
    #
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_information_path = saved_information_path + '/' + log_tag
    #
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_log_path = saved_information_path + '/Logs'
    #
    try:
        os.mkdir(saved_log_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_information_path = saved_information_path + '/' + save_model_name
    #
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_model_path = saved_information_path + '/trained_models'
    try:
        os.mkdir(saved_model_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    print('The current model is:')
    #
    print(save_model_name)
    #
    print('\n')
    #
    writer = SummaryWriter(saved_log_path + '/Log_' + save_model_name)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    start = timeit.default_timer()

    for epoch in range(num_epochs):

        model.train()

        train_h_dists = 0
        # train_f1 = 0
        train_iou = 0
        train_main_loss = 0
        # train_recall = 0
        # train_precision = 0
        train_effective_h = 0

        image_no_label = 0
        image_with_label = 0

        for j, (images1, images2, images3, labels, imagename) in enumerate(trainloader):

            # print(np.unique(labels))

            # print(np.unique(labels[0, :, :, :]))
            # print(np.unique(labels[1, :, :, :]))
            # print(np.unique(labels[2, :, :, :]))
            # print(np.unique(labels[3, :, :, :]))

            # overlayed = 0.4*np.squeeze(255*images2) + 0.6*np.squeeze(255*labels)
            # plt.imshow(overlayed, cmap='gray')
            # plt.show()

            # plt.imshow(np.squeeze(images2), cmap='gray')
            # plt.show()
            #
            # plt.imshow(np.squeeze(labels), cmap='gray')
            # plt.show()

            optimizer.zero_grad()
            images = images2.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)

            if 'ERF' in model_name:

                outputs_fp, outputs_fn = model(images)
                prob_outputs_fp = torch.sigmoid(outputs_fp)
                prob_outputs_fn = torch.sigmoid(outputs_fn)

                # check labels to see if they are :
                if torch.max(labels) == 100.0 and torch.min(labels) == 100.0:
                    loss = 0.0

                    train_main_loss += loss

                    image_no_label += 1

                else:
                    # print('Supervised')
                    image_with_label += 1
                    # images are with labels (0, 1)
                    if losstag == 'dice':

                        loss = dice_loss(prob_outputs_fp, labels) + dice_loss(prob_outputs_fn, labels)

                    elif losstag == 'ce':

                        loss = nn.BCEWithLogitsLoss(reduction='mean')(prob_outputs_fp, labels) + \
                               nn.BCEWithLogitsLoss(reduction='mean')(prob_outputs_fn, labels)

                    loss.backward()
                    optimizer.step()

                    train_main_loss += loss.item()

            elif 'Pertubation_inputs' in model_name:
                # (todo) I need to implement a baseline based on mean-teacher
                outputs1, outputs2 = model(images)
                prob_outputs = torch.sigmoid(outputs1) + torch.sigmoid(outputs2)
                prob_outputs = prob_outputs / 2

                if torch.max(labels) == 100.0 and torch.min(labels) == 100.0:

                    loss = 0.0

                    train_main_loss += loss

                    image_no_label += 1

                else:

                    image_with_label += 1

                    if losstag == 'dice':

                        loss = dice_loss(prob_outputs, labels)

                    elif losstag == 'ce':

                        loss = nn.BCEWithLogitsLoss(reduction='mean')(prob_outputs, labels)

                    loss.backward()
                    optimizer.step()

                    train_main_loss += loss.item()

            else:
                # fully supervised baseline

                # assert torch.max(labels) != 2.0

                outputs = model(images)
                prob_outputs = torch.sigmoid(outputs)

                if torch.max(labels) != 100.0:

                    image_with_label += 1

                    if losstag == 'dice':

                        loss = dice_loss(prob_outputs, labels)

                    elif losstag == 'ce':

                        loss = nn.BCEWithLogitsLoss(reduction='mean')(prob_outputs, labels)

                    loss.backward()
                    optimizer.step()

                    train_main_loss += loss.item()

                else:

                    loss = 0.0

                    train_main_loss += loss

                    image_no_label += 1

            # calculate training accuracy metric at the begining of each epoch:
            if j == 0:

                # calculate training accuracy at the end of each epoch
                if 'ERF' in model_name:
                    class_outputs = (prob_outputs_fn + prob_outputs_fp) / 2
                else:
                    class_outputs = prob_outputs

                # only calculate training accuracy when labels are available in training data:
                if torch.max(labels) != 100.0:

                    if (class_outputs == 1).sum() > 1 and (labels == 1).sum() > 1:

                        dist_ = hd95(class_outputs, labels, class_no)
                        train_h_dists += dist_
                        train_effective_h = train_effective_h + 1

                    train_mean_iu_ = segmentation_scores(labels, class_outputs, class_no)

                    # train_f1_, train_recall_, train_precision_, TPs_, TNs_, FPs_, FNs_, Ps_, Ns_ = f1_score(labels, class_outputs, class_no)

                    # train_f1 += train_f1_
                    train_iou += train_mean_iu_
                    # train_recall += train_recall_
                    # train_precision += train_precision_

                else:

                    train_mean_iu_ = 0.0
                    train_iou += train_mean_iu_

        # Evaluate at the end of each epoch:
        model.eval()
        with torch.no_grad():

            validate_iou = 0
            validate_f1 = 0
            validate_h_dist = 0
            validate_h_dist_effective = 0

            for i, (val_images1, val_images2, val_images3, val_label, imagename) in enumerate(validateloader):

                val_img = val_images2.to(device=device, dtype=torch.float32)
                val_label = val_label.to(device=device, dtype=torch.float32)

                assert torch.max(val_label) != 100.0
                # assert torch.min(val_label) == 0.0

                if 'ERF' in model_name:
                    #
                    val_outputs_fp, val_outputs_fn = model(val_img)
                    val_outputs_fp = torch.sigmoid(val_outputs_fp)
                    val_outputs_fn = torch.sigmoid(val_outputs_fn)
                    #
                    # inverse_val_outputs_fn = torch.ones_like(val_outputs_fn).to(device=device, dtype=torch.float32)
                    # inverse_val_outputs_fn = inverse_val_outputs_fn - val_outputs_fn
                    val_class_outputs = (val_outputs_fn + val_outputs_fp) / 2

                elif 'MTASSUnet' in model_name:

                    val_outputs, val_attention_outputs = model(val_img)
                    val_class_outputs = torch.sigmoid(val_outputs)

                elif 'Entropy_minimisation' in model_name:

                    val_outputs = model(val_img)
                    val_class_outputs = torch.sigmoid(val_outputs)

                else:

                    val_outputs = model(val_img)
                    val_class_outputs = torch.sigmoid(val_outputs)

                eval_mean_iu_ = segmentation_scores(val_label, val_class_outputs, class_no)
                eval_f1_, eval_recall_, eval_precision_, eTP, eTN, eFP, eFN, eP, eN = f1_score(val_label, val_class_outputs, class_no)
                validate_iou += eval_mean_iu_
                validate_f1 += eval_f1_
                #
                if (val_class_outputs == 1).sum() > 1 and (val_label == 1).sum() > 1:
                    v_dist_ = hd95(val_class_outputs, val_label, class_no)
                    validate_h_dist += v_dist_
                    validate_h_dist_effective = validate_h_dist_effective + 1

        print(
            'Step [{}/{}], '
            'Train main loss: {:.4f}, '
            'Train iou: {:.4f}, '
            'val iou:{:.4f}, '.format(epoch + 1, num_epochs,
                                      train_main_loss / (j + 1),
                                      train_iou / (image_with_label + 1),
                                      validate_iou / (i + 1)))

        # # # ================================================================== #
        # # #                        TensorboardX Logging                        #
        # # # # ================================================================ #
        writer.add_scalars('acc metrics', {'train iou': train_iou / (image_with_label + 1),
                                           'train hausdorff dist': train_h_dists / (train_effective_h+1),
                                           'val hausdorff dist': validate_h_dist / (validate_h_dist_effective + 1),
                                           'val iou': validate_iou / (i + 1),
                                           'val f1': validate_f1 / (i + 1)}, epoch + 1)

        writer.add_scalars('loss values', {'main loss': train_main_loss / (j+1)}, epoch + 1)

        # for param_group in optimizer.param_groups:
        #
        #     if epoch == (num_epochs - 10):
        #         param_group['lr'] = learning_rate * 0.1
        #         # param_group['lr'] = learning_rate * ((1 - epoch / num_epochs) ** 0.999)

        if epoch >= (num_epochs//2):

            save_model_name_full = saved_model_path + '/' + save_model_name + '_epoch' + str(epoch) + '.pt'

            path_model = save_model_name_full

            torch.save(model, path_model)

    save_model_name_full = saved_model_path + '/' + save_model_name + '_Final.pt'

    path_model = save_model_name_full

    torch.save(model, path_model)

    # Testing:
    save_path = saved_information_path + '/Visual_results'

    try:

        os.mkdir(save_path)

    except OSError as exc:

        if exc.errno != errno.EEXIST:

            raise

        pass

    all_models = glob.glob(os.path.join(saved_model_path, '*.pt'))

    test_iou = []
    test_f1 = []
    test_h_dist = []

    for model in all_models:

        model = torch.load(model)
        model.eval()

        with torch.no_grad():

            for ii, (test_images1, test_images2, test_images3, test_label, test_imagename) in enumerate(testdata):

                test_img = test_images2.to(device=device, dtype=torch.float32)
                test_label = test_label.to(device=device, dtype=torch.float32)

                assert torch.max(test_label) != 100.0

                if 'ERF' in model_name:

                    test_outputs_fp, test_outputs_fn = model(test_img)
                    test_outputs_fp = torch.sigmoid(test_outputs_fp)
                    test_outputs_fn = torch.sigmoid(test_outputs_fn)

                    # inverse_test_outputs_fn = torch.ones_like(test_outputs_fn).to(device=device, dtype=torch.float32)
                    # inverse_test_outputs_fn = inverse_test_outputs_fn - test_outputs_fn
                    test_class_outputs = (test_outputs_fn + test_outputs_fp) / 2

                elif 'MTASSUnet' in model_name:

                    test_outputs, test_attention_outputs = model(test_img)
                    test_class_outputs = torch.sigmoid(test_outputs)

                elif 'Entropy_minimisation' in model_name:

                    test_outputs = model(test_img)
                    test_class_outputs = torch.sigmoid(test_outputs)

                else:
                    test_outputs = model(test_img)
                    test_class_outputs = torch.sigmoid(test_outputs)

                test_mean_iu_ = segmentation_scores(test_label, test_class_outputs, class_no)
                test_f1_, test_recall_, test_precision_, eTP, eTN, eFP, eFN, eP, eN = f1_score(test_label, test_class_outputs, class_no)

                test_iou.append(test_mean_iu_)
                test_f1.append(test_f1_)
                # test_iou += test_mean_iu_
                # test_f1 += test_f1_

                if (test_class_outputs == 1).sum() > 1 and (test_label == 1).sum() > 1:
                    t_dist_ = hd95(test_class_outputs, test_label, class_no)
                    test_h_dist.append(t_dist_)
                    # test_h_dist += t_dist_
                    # test_h_dist_effective = test_h_dist_effective + 1

                # save segmentation:
                if save_all_segmentation is True:
                    save_name = save_path + '/test_' + str(ii) + '_seg.png'
                    save_name_label = save_path + '/test_' + str(ii) + '_label.png'
                    (b, c, h, w) = test_label.shape
                    assert c == 1
                    test_class_outputs = test_class_outputs.reshape(h, w).cpu().detach().numpy() > 0.5
                    plt.imsave(save_name, test_class_outputs, cmap='gray')
                    plt.imsave(save_name_label, test_label.reshape(h, w).cpu().detach().numpy(), cmap='gray')

    result_dictionary = {
        'Test IoU mean': str(np.mean(test_iou)),
        'Test IoU std': str(np.std(test_iou)),
        'Test f1 mean': str(np.mean(test_f1)),
        'Test f1 std': str(np.std(test_f1)),
        'Test H-dist mean': str(np.mean(test_h_dist)),
        'Test H-dist std': str(np.std(test_h_dist))
    }

    ff_path = save_path + '/test_result_data.txt'
    ff = open(ff_path, 'w')
    ff.write(str(result_dictionary))
    ff.close()

    # save model
    stop = timeit.default_timer()

    print('Time: ', stop - start)

    print('\nTraining finished and model saved\n')

    print('Test IoU: ' + str(np.nanmean(test_iou)) + '\n')
    print('Test H-dist: ' + str(np.nanmean(test_h_dist)) + '\n')
    print('Test IoU std: ' + str(np.nanstd(test_iou)) + '\n')
    print('Test H-dist std: ' + str(np.nanstd(test_h_dist)) + '\n')

    return model

