import os
import gzip
import errno
import shutil
import random
# import pydicom
import numpy as np
from PIL import Image
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import resize
# from nipype.interfaces.ants import N4BiasFieldCorrection
from tifffile import imsave

# extract slices from each case


def generate_slices_per_case(
    data,
    training_slices_no,
    save_folder,
    new_size,
    tag_class,
    foreground_threshold
    ):

    # prepare saving locations
    case_patches_folder = save_folder + '/patches/'
    case_labels_folder = save_folder + '/labels/'

    try:
        os.makedirs(case_patches_folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    pass

    try:
        os.makedirs(case_labels_folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    pass

    for item in data:

        all_sub_folders = os.listdir(item)
        all_modalities = [os.path.join(item, x) for index, x in enumerate(all_sub_folders) if 'seg' not in x]

        all_modalities.sort()
        gt_path = [os.path.join(item, x) for index, x in enumerate(all_sub_folders) if 'seg' in x]

        # read all modalities for case for validation
        t1 = nib.load(all_modalities[0])
        t1 = t1.get_fdata()
        t2 = nib.load(all_modalities[1])
        t2 = t2.get_fdata()
        t1ce = nib.load(all_modalities[2])
        t1ce = t1ce.get_fdata()
        flair = nib.load(all_modalities[3])
        flair = flair.get_fdata()
        # normalise based on all non-zero elements:
        t1_non_zero = t1[np.nonzero(t1)]
        t2_non_zero = t2[np.nonzero(t2)]
        t1ce_non_zero = t1ce[np.nonzero(t1ce)]
        flair_non_zero = flair[np.nonzero(flair)]

        t1 = (t1 - t1_non_zero.mean()) / t1_non_zero.std()
        t2 = (t2 - t2_non_zero.mean()) / t2_non_zero.std()
        t1ce = (t1ce - t1ce_non_zero.mean()) / t1ce_non_zero.std()
        flair = (flair - flair_non_zero.mean()) / flair_non_zero.std()
        # ground truth of lgg case for validation:
        gt_path = gt_path[0]
        gt = nib.load(gt_path)
        gt = gt.get_fdata()

        if tag_class == 'WT':
            gt[gt == 0] = 0
            gt[gt == 1] = 1
            gt[gt == 2] = 1
            # gt[gt == 3] = 0
            gt[gt == 4] = 1
        elif tag_class == 'ET':
            gt[gt == 0] = 0
            gt[gt == 1] = 0
            gt[gt == 2] = 0
            # gt[gt == 3] = 0
            gt[gt == 4] = 1
        elif tag_class == 'TC':
            gt[gt == 0] = 0
            gt[gt == 1] = 1
            gt[gt == 2] = 0
            # gt[gt == 3] = 0
            gt[gt == 4] = 1
        elif tag_class == 'All':
            gt[gt == 0] = 0
            gt[gt == 1] = 1
            gt[gt == 2] = 2
            gt[gt == 4] = 3
        elif tag_class == 'No':
            gt[gt == 0] = 100
            gt[gt == 1] = 100
            gt[gt == 2] = 100
            gt[gt == 4] = 100

        height, width, channels = gt.shape

        if training_slices_no < channels:
            last_index_slice = training_slices_no
        else:
            last_index_slice = channels - 2

        # extract case number and name:
        fullfilename, extenstion = os.path.splitext(gt_path)
        dirpath_parts = fullfilename.split('/')
        case_index = dirpath_parts[-1]

        first_slice_over_threshold = 0

        if training_slices_no < channels:

            for i in range(channels):

                gt_slice = gt[:, :, i]
                all_tumours = np.sum(gt_slice)
                if all_tumours > foreground_threshold:
                    break
                first_slice_over_threshold = first_slice_over_threshold + 1

            print('first slice with tumour above threshold:' + str(first_slice_over_threshold))

        for i in range(last_index_slice):

            if training_slices_no < channels:
                no = i + first_slice_over_threshold
            else:
                no = i + 1

            gt_slice_store_name = case_index + '_gt_' + str(no) + '.tif'
            img_slice_store_name = case_index + '_slice_' + str(no) + '.tif'

            label_store_path_full = os.path.join(
                case_labels_folder, gt_slice_store_name)
            patch_store_path_full = os.path.join(
                case_patches_folder, img_slice_store_name)

            gt_slice = gt[:, :, no]

            # plt.imshow(gt_slice, cmap='gray')
            # plt.show()
            # print('tumour pixels:' + str(np.sum(gt_slice)))

            h, w = np.shape(gt_slice)
            gt_slice = np.asarray(gt_slice, dtype=np.float32)
            left = int(np.ceil((w - new_size) / 2))
            right = w - int(np.floor((w - new_size) / 2))
            top = int(np.ceil((h - new_size) / 2))
            bottom = h - int(np.floor((h - new_size) / 2))
            gt_slice = gt_slice[top:bottom, left:right]

            t1_slice = t1[:, :, no-1:no+2]
            # print(np.shape(t1_slice))
            t1_slice = np.asarray(t1_slice, dtype=np.float32)
            t1_slice = t1_slice[top:bottom, left:right, :]
            # print(np.shape(t1_slice))
            t1_slice = np.transpose(t1_slice, (2, 0, 1))
            # plt.imshow(t1_slice[0, :, :], cmap='gray')
            # plt.show()

            # print(np.shape(t1_slice))
            # print(np.shape(t1_slice))
            # t1_slice = np.reshape(t1_slice, (3, new_size, new_size))

            t2_slice = t2[:, :, no-1:no+2]
            t2_slice = np.asarray(t2_slice, dtype=np.float32)
            t2_slice = t2_slice[top:bottom, left:right, :]
            t2_slice = np.transpose(t2_slice, (2, 0, 1))
            # t2_slice = np.reshape(t2_slice, (1, new_size, new_size))

            t1ce_slice = t1ce[:, :, no-1:no+2]
            t1ce_slice = np.asarray(t1ce_slice, dtype=np.float32)
            t1ce_slice = t1ce_slice[top:bottom, left:right, :]
            t1ce_slice = np.transpose(t1ce_slice, (2, 0, 1))
            # t1ce_slice = np.reshape(t1ce_slice, (1, new_size, new_size))

            flair_slice = flair[:, :, no-1:no+2]
            flair_slice = np.asarray(flair_slice, dtype=np.float32)
            flair_slice = flair_slice[top:bottom, left:right, :]
            flair_slice = np.transpose(flair_slice, (2, 0, 1))
            # flair_slice = np.reshape(flair_slice, (1, new_size, new_size))

            multi_modal_slice = t1_slice
            multi_modal_slice = np.concatenate(
                (multi_modal_slice, t2_slice), axis=0)
            multi_modal_slice = np.concatenate(
                (multi_modal_slice, t1ce_slice), axis=0)
            multi_modal_slice = np.concatenate(
                (multi_modal_slice, flair_slice), axis=0)

            # print(np.shape(multi_modal_slice))

            # non_zero_gt = np.count_nonzero(gt_slice)
            non_zero_gt = np.sum(gt_slice)

            if non_zero_gt > foreground_threshold:
                print(np.shape(multi_modal_slice))
                imsave(patch_store_path_full, multi_modal_slice)
                print(img_slice_store_name + '_' + tag_class + ' saved')
                imsave(label_store_path_full, gt_slice)
                print(gt_slice_store_name + '_' + tag_class + ' saved')
                print('tumour area of this slice is:' + str(non_zero_gt))
                print('\n')


def main_run(data_folder,
             cases_labelled_train,
             cases_unlabelled_train,
             cases_val,
             cases_test,
             save_folder,
             tag_class,
             training_slices_no,
             foreground_threshold,
             new_size
             ):

    save_folder = save_folder + '/t' + str(foreground_threshold) + '_s' + str(training_slices_no) + '_l' + str(cases_labelled_train) + '_u' + str(cases_unlabelled_train) + '_' + tag_class

    try:
        os.makedirs(save_folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    pass

    all_data = os.listdir(data_folder)
    all_data = [os.path.join(data_folder, x) for x in all_data]

    if cases_labelled_train == 1:
        train_case_labelled = [all_data[0]]
    else:
        train_case_labelled = all_data[0:cases_labelled_train]

    train_case_unlabelled = all_data[cases_labelled_train:cases_labelled_train+cases_unlabelled_train]
    validate_case = all_data[cases_labelled_train+cases_unlabelled_train:cases_labelled_train+cases_unlabelled_train+cases_val]
    test_case = all_data[cases_labelled_train+cases_unlabelled_train+cases_val:cases_labelled_train+cases_unlabelled_train+cases_val+cases_test]

    # labelled training data:
    save_folder_train = save_folder + '/train'
    try:
        os.makedirs(save_folder_train)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    pass
    generate_slices_per_case(train_case_labelled, training_slices_no, save_folder_train, new_size, tag_class, foreground_threshold)
    # unlabelled training data
    generate_slices_per_case(train_case_unlabelled, 10000, save_folder_train, new_size, 'No', foreground_threshold)
    # validate data
    save_folder_val = save_folder + '/validate'
    try:
        os.makedirs(save_folder_val)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    pass

    generate_slices_per_case(validate_case, 10000, save_folder_val, new_size, tag_class, foreground_threshold)
    # test data
    save_folder_test = save_folder + '/test'
    try:
        os.makedirs(save_folder_test)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    pass
    generate_slices_per_case(test_case, 10000, save_folder_test, new_size, tag_class, foreground_threshold)


if __name__ == '__main__':

    # change here:
    cases_labelled_train = 1
    cases_unlabelled_train = 40
    tag_class = 'WT'
    training_slices_no = 20
    foreground_threshold = 5
    # =====================
    cases_val = 2
    cases_test = 40

    data_folder = '/home/moucheng/projects_data/Brain_data/brats2018/HGG'
    save_folder = '/home/moucheng/projects_data/Brain_data/brats2018'

    # data_folder = '/home/moucheng/projects_data/brain/BRATS2018/HGG'
    # save_folder = '/home/moucheng/projects_data/brain/BRATS2018'

    main_run(
        data_folder,
        cases_labelled_train,
        cases_unlabelled_train,
        cases_val,
        cases_test,
        save_folder,
        tag_class,
        training_slices_no,
        foreground_threshold,
        new_size=176,
    )

print('End')






