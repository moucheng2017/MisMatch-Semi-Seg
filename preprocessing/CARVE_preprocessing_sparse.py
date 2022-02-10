import SimpleITK as sitk

import random
import numpy as np
import os
import shutil

import errno

from distutils.dir_util import copy_tree

from tifffile import imsave


def chunks(l, n):
    # l: the whole list to be divided
    # n: amount of elements for each subgroup
    # Yield successive n-sized chunks from l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return ct_scan, origin, spacing


def prepare_data(new_size, scans_folder, labels_folder, save_folder, input_dim, train_amount_slices, case_no):
    ''' This cropped every corners
    :param new_size:
    :param scans_folder:
    :param labels_folder:
    :param save_folder:
    :param input_dim:
    :param training_amount_slices: number of slices for each case used for training with labels
    :param training_without_labels_case: number of cases used for training without labels
    :return:
    '''
    # # padding:
    # left = int(np.ceil((new_size - width) / 2))
    # right = new_size - int(np.floor((new_size - width) / 2))
    # top = int(np.ceil((new_size - height) / 2))
    # bottom = new_size - int(np.floor((new_size - height) / 2))
    # ===============================================
    # 1. read all scans
    # 2. divide scans into train, validate and test
    # 3. divide train into labels and without labels:
    # 4. save for each sub dataset:
    # ===============================================
    all_cases_scans = [x for x in os.listdir(scans_folder) if 'mhd' in x]
    all_cases_scans.sort()
    all_cases_labels = [x for x in os.listdir(labels_folder) if 'mhd' in x]
    all_cases_labels.sort()

    all_cases_scans = [os.path.join(scans_folder, x) for x in all_cases_scans]
    all_cases_labels = [os.path.join(labels_folder, x) for x in all_cases_labels]

    train_withlabels_scans = [all_cases_scans[case_no]]
    train_withlabels_labels = [all_cases_labels[case_no]]

    # rest_cases_scans = set(all_cases_scans).difference(train_withlabels_scans)
    # rest_cases_labels = set(all_cases_labels).difference(train_withlabels_labels)

    rest_cases_scans = [x for x in all_cases_scans if x not in train_withlabels_scans]
    rest_cases_labels = [x for x in all_cases_labels if x not in train_withlabels_labels]

    # print(rest_cases_scans)

    train_withoutlabels_scans = rest_cases_scans[0:2]
    train_withoutlabels_labels = rest_cases_labels[0:2]

    validate_scans = [rest_cases_scans[3]]
    validate_labels = [rest_cases_labels[3]]

    test_scans = rest_cases_scans[4:8]
    test_labels = rest_cases_labels[4:8]

    if input_dim == 3:
        save_lo = save_folder + '/sparse_' + str(case_no) + '_r' + str(new_size) + '_s' + str(train_amount_slices)
    elif input_dim == 1:
        save_lo = save_folder + '/sparse_' + str(case_no) + '_r' + str(new_size) + '_s' + str(train_amount_slices)
    elif input_dim == 5:
        save_lo = save_folder + '/sparse_' + str(case_no) + '_r' + str(new_size) + '_s' + str(train_amount_slices)

    validate_folder_patches = os.path.join(save_lo, 'validate/patches')
    validate_folder_labels = os.path.join(save_lo, 'validate/labels')
    validate_save_folder = os.path.join(save_lo, 'validate')

    train_folder_patches = os.path.join(save_lo, 'train/patches')
    train_folder_labels = os.path.join(save_lo, 'train/labels')
    train_save_folder = os.path.join(save_lo, 'train')
    # train_folder_with_labels_patches = os.path.join(save_lo, 'train/with_labels/patches')
    # train_folder_with_labels_labels = os.path.join(save_lo, 'train/with_labels/labels')

    # train_folder_without_labels_patches = os.path.join(save_lo, 'train/without_labels/patches')
    # train_folder_without_labels_labels = os.path.join(save_lo, 'train/without_labels/labels')

    # train_save_folder = os.path.join(save_lo, 'train')
    # train_with_labels_save_folder = os.path.join(save_lo, 'train/with_labels')
    # train_without_labels_save_folder = os.path.join(save_lo, 'train/without_labels')

    test_folder_patches = os.path.join(save_lo, 'test/patches')
    test_folder_labels = os.path.join(save_lo, 'test/labels')
    test_save_folder = os.path.join(save_lo, 'test')

    try:
        os.makedirs(save_lo)
        os.makedirs(train_save_folder)
        os.makedirs(test_save_folder)
        os.makedirs(validate_save_folder)

        os.makedirs(train_folder_patches)
        os.makedirs(train_folder_labels)

        os.makedirs(validate_folder_patches)
        os.makedirs(validate_folder_labels)

        # os.makedirs(train_folder_with_labels_patches)
        # os.makedirs(train_folder_with_labels_labels)
        #
        # os.makedirs(train_folder_without_labels_patches)
        # os.makedirs(train_folder_without_labels_labels)
        #
        # os.makedirs(train_without_labels_save_folder)
        # os.makedirs(train_with_labels_save_folder)

        os.makedirs(test_folder_patches)
        os.makedirs(test_folder_labels)

    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    pass

    if train_amount_slices == 10000:

        pre_process(new_size=new_size, all_cases_scans=train_withoutlabels_scans, all_cases_labels=train_withoutlabels_labels, save_folder=train_save_folder, input_dim=input_dim, train_amount_slices=10000, with_label=True, random_sample=True)
        pre_process(new_size=new_size, all_cases_scans=train_withlabels_scans, all_cases_labels=train_withlabels_labels, save_folder=train_save_folder, input_dim=input_dim, train_amount_slices=10000, with_label=True, random_sample=True)

    else:

        pre_process(new_size=new_size, all_cases_scans=train_withoutlabels_scans, all_cases_labels=train_withoutlabels_labels, save_folder=train_save_folder, input_dim=input_dim, train_amount_slices=10000, with_label=False, random_sample=True)
        pre_process(new_size=new_size, all_cases_scans=train_withlabels_scans, all_cases_labels=train_withlabels_labels, save_folder=train_save_folder, input_dim=input_dim, train_amount_slices=train_amount_slices, with_label=True, random_sample=True)
    #
    pre_process(new_size=new_size, all_cases_scans=validate_scans, all_cases_labels=validate_labels, save_folder=validate_save_folder, input_dim=input_dim, train_amount_slices=10000, with_label=True, random_sample=True)
    pre_process(new_size=new_size, all_cases_scans=test_scans, all_cases_labels=test_labels, save_folder=test_save_folder, input_dim=input_dim, train_amount_slices=10000, with_label=True, random_sample=True)


def pre_process(new_size, all_cases_scans, all_cases_labels, save_folder, input_dim, train_amount_slices, with_label, random_sample):

    # save_folder = os.path.join(save_folder, str(new_size))
    # all_cases_scans = [x for x in os.listdir(scans_folder) if 'mhd' in x]
    # all_cases_scans.sort()
    # all_cases_labels = [x for x in os.listdir(labels_folder) if 'mhd' in x]
    # all_cases_labels.sort()
    # case_number = 0

    # print(all_cases_scans)

    for case_scan, case_label in zip(all_cases_scans, all_cases_labels):

        # case_scan = os.path.join(scans_folder, case_scan)
        # case_label = os.path.join(labels_folder, case_label)

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

        # case_number = case_number + 1

        fullfilename, extenstion = os.path.splitext(case_scan)

        dirpath_parts = fullfilename.split('/')
        case_index = dirpath_parts[-1]

        # print(case_index)

        ct_scan, ct_origin, ct_spacing = load_itk(case_scan)
        label_scan, label_origin, label_spacing = load_itk(case_label)
        slices_amount, height, width = np.shape(ct_scan)

        print('\n')
        print('The current case is:' + case_index)
        print('slice amount:' + str(slices_amount))
        print('slice height:' + str(height))
        print('slice width:' + str(width))

        ct_scan = ct_scan.astype(np.float32)
        ct_scan[ct_scan < -950.0] = -950.0
        ct_scan[ct_scan > 100.0] = 100.0
        ct_scan = (ct_scan - np.nanmean(ct_scan)) / np.nanstd(ct_scan)

        label_scan = label_scan.astype(np.int8)

        if train_amount_slices > slices_amount:
            # 10000
            last_slice_no = slices_amount - 5
        else:
            last_slice_no = train_amount_slices

        if with_label is True:

            label_scan[label_scan == 1] = 1
            label_scan[label_scan == 2] = 1
            label_scan[label_scan != 1] = 0

        else:

            label_scan[label_scan == 1] = 100
            label_scan[label_scan == 2] = 100
            label_scan[label_scan != 1] = 100

        # if train_amount_slices > slices_amount:

        for slice_no in range(slices_amount):

            temp_label_slice_ = label_scan[slice_no, :, :]

            # label_max = np.max(temp_label_slice_)

            all_vessels = np.sum(temp_label_slice_)

            # print(label_max)

            if all_vessels > 100:

                break

        first_slice_with_vessel = slice_no

        print('First vessel appear at:' + str(first_slice_with_vessel))

        if random_sample is False:

            for slice_no in range(last_slice_no):

                # position_dice = random.uniform(0, 1)

                if train_amount_slices > slices_amount:

                    sample_position = slice_no

                else:

                    sample_position = slice_no + first_slice_with_vessel

                if input_dim == 1:
                    temp_scan_slice_2_ = ct_scan[sample_position, :, :]
                elif input_dim == 3:
                    temp_scan_slice_1_ = ct_scan[sample_position - 1, :, :]
                    temp_scan_slice_2_ = ct_scan[sample_position, :, :]
                    temp_scan_slice_3_ = ct_scan[sample_position + 1, :, :]
                elif input_dim == 5:
                    temp_scan_slice_0_ = ct_scan[sample_position - 2, :, :]
                    temp_scan_slice_1_ = ct_scan[sample_position - 1, :, :]
                    temp_scan_slice_2_ = ct_scan[sample_position, :, :]
                    temp_scan_slice_3_ = ct_scan[sample_position + 1, :, :]
                    temp_scan_slice_4_ = ct_scan[sample_position + 2, :, :]

                temp_label_slice_ = label_scan[sample_position, :, :]

                # print(temp_label_slice_.shape)

                # height, width = temp_scan_slice_1.shape

                height, width = temp_label_slice_.shape

                # if height >= new_size and width >= new_size:
                assert height >= new_size
                assert width >= new_size
                # ======================
                # cropping from left up:
                # ======================
                # if position_dice <= 0.25:
                #
                #     left = 0
                #     right = new_size
                #     top = 0
                #     bottom = new_size
                #
                # elif position_dice <= 0.5:
                #
                #     left = width - new_size
                #     right = width
                #     top = 0
                #     bottom = new_size
                #
                # elif position_dice <= 0.75:
                #
                #     left = 0
                #     right = new_size
                #     top = height - new_size
                #     bottom = height
                #
                # else:
                #
                #     left = width - new_size
                #     right = width
                #     top = height - new_size
                #     bottom = height

                # =================================
                left = 0
                right = new_size
                top = 0
                bottom = new_size

                if input_dim == 5:

                    temp_scan_slice_0 = temp_scan_slice_0_[top:bottom, left:right]
                    temp_scan_slice_1 = temp_scan_slice_1_[top:bottom, left:right]
                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]
                    temp_scan_slice_3 = temp_scan_slice_3_[top:bottom, left:right]
                    temp_scan_slice_4 = temp_scan_slice_4_[top:bottom, left:right]

                    temp_scan_slice_0 = np.reshape(temp_scan_slice_0, (1, new_size, new_size))
                    temp_scan_slice_1 = np.reshape(temp_scan_slice_1, (1, new_size, new_size))
                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))
                    temp_scan_slice_3 = np.reshape(temp_scan_slice_3, (1, new_size, new_size))
                    temp_scan_slice_4 = np.reshape(temp_scan_slice_4, (1, new_size, new_size))

                    scan = np.concatenate((temp_scan_slice_0, temp_scan_slice_1, temp_scan_slice_2, temp_scan_slice_3, temp_scan_slice_4), axis=0)

                elif input_dim == 3:

                    temp_scan_slice_1 = temp_scan_slice_1_[top:bottom, left:right]
                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]
                    temp_scan_slice_3 = temp_scan_slice_3_[top:bottom, left:right]

                    temp_scan_slice_1 = np.reshape(temp_scan_slice_1, (1, new_size, new_size))
                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))
                    temp_scan_slice_3 = np.reshape(temp_scan_slice_3, (1, new_size, new_size))

                    scan = np.concatenate((temp_scan_slice_1, temp_scan_slice_2, temp_scan_slice_3), axis=0)

                elif input_dim == 1:

                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]

                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))

                    scan = temp_scan_slice_2

                save_img_name = case_index + '_1_' + str(slice_no) + '_slice.tif'
                full_save_img_name = os.path.join(case_patches_folder, save_img_name)
                #
                label = temp_label_slice_[top:bottom, left:right]
                save_label_name = case_index + '_1_' + str(slice_no) + '_label.tif'
                full_save_label_name = os.path.join(case_labels_folder, save_label_name)

                #
                # print(label.max())
                #
                # dim, new_height, new_width = np.shape(scan)
                new_height, new_width = np.shape(label)

                # print(np.shape(scan))

                assert new_height == new_size
                assert new_width == new_size
                # if new_size != new_new_size:
                #     scan = resize(scan, (new_new_size, new_new_size))
                # scan = Image.fromarray(scan, mode='F')  # float32
                imsave(full_save_img_name, scan)
                # scan.save(full_save_img_name, "TIFF")
                new_height, new_width = np.shape(label)

                assert new_height == new_size
                assert new_width == new_size
                # if new_size != new_new_size:
                #     label = resize(label, (new_new_size, new_new_size))
                #     label.round()
                # label = Image.fromarray(label)  # float32
                imsave(full_save_label_name, label)
                # label.save(full_save_label_name, "TIFF")

                # =================================
                left = width - new_size
                right = width
                top = 0
                bottom = new_size

                if input_dim == 5:

                    temp_scan_slice_0 = temp_scan_slice_0_[top:bottom, left:right]
                    temp_scan_slice_1 = temp_scan_slice_1_[top:bottom, left:right]
                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]
                    temp_scan_slice_3 = temp_scan_slice_3_[top:bottom, left:right]
                    temp_scan_slice_4 = temp_scan_slice_4_[top:bottom, left:right]

                    temp_scan_slice_0 = np.reshape(temp_scan_slice_0, (1, new_size, new_size))
                    temp_scan_slice_1 = np.reshape(temp_scan_slice_1, (1, new_size, new_size))
                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))
                    temp_scan_slice_3 = np.reshape(temp_scan_slice_3, (1, new_size, new_size))
                    temp_scan_slice_4 = np.reshape(temp_scan_slice_4, (1, new_size, new_size))

                    scan = np.concatenate((temp_scan_slice_0, temp_scan_slice_1, temp_scan_slice_2, temp_scan_slice_3, temp_scan_slice_4), axis=0)

                elif input_dim == 3:

                    temp_scan_slice_1 = temp_scan_slice_1_[top:bottom, left:right]
                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]
                    temp_scan_slice_3 = temp_scan_slice_3_[top:bottom, left:right]

                    temp_scan_slice_1 = np.reshape(temp_scan_slice_1, (1, new_size, new_size))
                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))
                    temp_scan_slice_3 = np.reshape(temp_scan_slice_3, (1, new_size, new_size))

                    scan = np.concatenate((temp_scan_slice_1, temp_scan_slice_2, temp_scan_slice_3), axis=0)

                elif input_dim == 1:

                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]

                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))

                    scan = temp_scan_slice_2

                save_img_name = case_index + '_2_' + str(slice_no) + '_slice.tif'
                full_save_img_name = os.path.join(case_patches_folder, save_img_name)
                #
                label = temp_label_slice_[top:bottom, left:right]
                save_label_name = case_index + '_2_' + str(slice_no) + '_label.tif'
                full_save_label_name = os.path.join(case_labels_folder, save_label_name)

                #
                # print(label.max())
                #
                # dim, new_height, new_width = np.shape(scan)
                new_height, new_width = np.shape(label)

                # print(np.shape(scan))

                assert new_height == new_size
                assert new_width == new_size
                # if new_size != new_new_size:
                #     scan = resize(scan, (new_new_size, new_new_size))
                # scan = Image.fromarray(scan, mode='F')  # float32
                imsave(full_save_img_name, scan)
                # scan.save(full_save_img_name, "TIFF")
                new_height, new_width = np.shape(label)

                assert new_height == new_size
                assert new_width == new_size
                # if new_size != new_new_size:
                #     label = resize(label, (new_new_size, new_new_size))
                #     label.round()
                # label = Image.fromarray(label)  # float32
                imsave(full_save_label_name, label)
                # label.save(full_save_label_name, "TIFF")

                # =================================
                left = 0
                right = new_size
                top = height - new_size
                bottom = height

                if input_dim == 5:

                    temp_scan_slice_0 = temp_scan_slice_0_[top:bottom, left:right]
                    temp_scan_slice_1 = temp_scan_slice_1_[top:bottom, left:right]
                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]
                    temp_scan_slice_3 = temp_scan_slice_3_[top:bottom, left:right]
                    temp_scan_slice_4 = temp_scan_slice_4_[top:bottom, left:right]

                    temp_scan_slice_0 = np.reshape(temp_scan_slice_0, (1, new_size, new_size))
                    temp_scan_slice_1 = np.reshape(temp_scan_slice_1, (1, new_size, new_size))
                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))
                    temp_scan_slice_3 = np.reshape(temp_scan_slice_3, (1, new_size, new_size))
                    temp_scan_slice_4 = np.reshape(temp_scan_slice_4, (1, new_size, new_size))

                    scan = np.concatenate((temp_scan_slice_0, temp_scan_slice_1, temp_scan_slice_2, temp_scan_slice_3, temp_scan_slice_4), axis=0)

                elif input_dim == 3:

                    temp_scan_slice_1 = temp_scan_slice_1_[top:bottom, left:right]
                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]
                    temp_scan_slice_3 = temp_scan_slice_3_[top:bottom, left:right]

                    temp_scan_slice_1 = np.reshape(temp_scan_slice_1, (1, new_size, new_size))
                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))
                    temp_scan_slice_3 = np.reshape(temp_scan_slice_3, (1, new_size, new_size))

                    scan = np.concatenate((temp_scan_slice_1, temp_scan_slice_2, temp_scan_slice_3), axis=0)

                elif input_dim == 1:

                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]

                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))

                    scan = temp_scan_slice_2

                save_img_name = case_index + '_3_' + str(slice_no) + '_slice.tif'
                full_save_img_name = os.path.join(case_patches_folder, save_img_name)
                #
                label = temp_label_slice_[top:bottom, left:right]
                save_label_name = case_index + '_3_' + str(slice_no) + '_label.tif'
                full_save_label_name = os.path.join(case_labels_folder, save_label_name)

                #
                # print(label.max())
                #
                # dim, new_height, new_width = np.shape(scan)
                new_height, new_width = np.shape(label)

                # print(np.shape(scan))

                assert new_height == new_size
                assert new_width == new_size
                # if new_size != new_new_size:
                #     scan = resize(scan, (new_new_size, new_new_size))
                # scan = Image.fromarray(scan, mode='F')  # float32
                imsave(full_save_img_name, scan)
                # scan.save(full_save_img_name, "TIFF")
                new_height, new_width = np.shape(label)

                assert new_height == new_size
                assert new_width == new_size
                # if new_size != new_new_size:
                #     label = resize(label, (new_new_size, new_new_size))
                #     label.round()
                # label = Image.fromarray(label)  # float32
                imsave(full_save_label_name, label)
                # label.save(full_save_label_name, "TIFF")

                # =================================
                left = width - new_size
                right = width
                top = height - new_size
                bottom = height

                if input_dim == 5:

                    temp_scan_slice_0 = temp_scan_slice_0_[top:bottom, left:right]
                    temp_scan_slice_1 = temp_scan_slice_1_[top:bottom, left:right]
                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]
                    temp_scan_slice_3 = temp_scan_slice_3_[top:bottom, left:right]
                    temp_scan_slice_4 = temp_scan_slice_4_[top:bottom, left:right]

                    temp_scan_slice_0 = np.reshape(temp_scan_slice_0, (1, new_size, new_size))
                    temp_scan_slice_1 = np.reshape(temp_scan_slice_1, (1, new_size, new_size))
                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))
                    temp_scan_slice_3 = np.reshape(temp_scan_slice_3, (1, new_size, new_size))
                    temp_scan_slice_4 = np.reshape(temp_scan_slice_4, (1, new_size, new_size))

                    scan = np.concatenate((temp_scan_slice_0, temp_scan_slice_1, temp_scan_slice_2, temp_scan_slice_3, temp_scan_slice_4), axis=0)

                elif input_dim == 3:

                    temp_scan_slice_1 = temp_scan_slice_1_[top:bottom, left:right]
                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]
                    temp_scan_slice_3 = temp_scan_slice_3_[top:bottom, left:right]

                    temp_scan_slice_1 = np.reshape(temp_scan_slice_1, (1, new_size, new_size))
                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))
                    temp_scan_slice_3 = np.reshape(temp_scan_slice_3, (1, new_size, new_size))

                    scan = np.concatenate((temp_scan_slice_1, temp_scan_slice_2, temp_scan_slice_3), axis=0)

                elif input_dim == 1:

                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]

                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))

                    scan = temp_scan_slice_2

                save_img_name = case_index + '_4_' + str(slice_no) + '_slice.tif'
                full_save_img_name = os.path.join(case_patches_folder, save_img_name)
                #
                label = temp_label_slice_[top:bottom, left:right]
                save_label_name = case_index + '_4_' + str(slice_no) + '_label.tif'
                full_save_label_name = os.path.join(case_labels_folder, save_label_name)

                #
                # print(label.max())
                #
                # dim, new_height, new_width = np.shape(scan)
                new_height, new_width = np.shape(label)

                # print(np.shape(scan))

                assert new_height == new_size
                assert new_width == new_size
                # if new_size != new_new_size:
                #     scan = resize(scan, (new_new_size, new_new_size))
                # scan = Image.fromarray(scan, mode='F')  # float32
                imsave(full_save_img_name, scan)
                # scan.save(full_save_img_name, "TIFF")
                new_height, new_width = np.shape(label)

                assert new_height == new_size
                assert new_width == new_size
                # if new_size != new_new_size:
                #     label = resize(label, (new_new_size, new_new_size))
                #     label.round()
                # label = Image.fromarray(label)  # float32
                imsave(full_save_label_name, label)
                # label.save(full_save_label_name, "TIFF")

                # =================================
                left = (width - new_size)//2
                right = left + new_size
                top = (height - new_size)//2
                bottom = (height - new_size)//2 + new_size

                if input_dim == 5:

                    temp_scan_slice_0 = temp_scan_slice_0_[top:bottom, left:right]
                    temp_scan_slice_1 = temp_scan_slice_1_[top:bottom, left:right]
                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]
                    temp_scan_slice_3 = temp_scan_slice_3_[top:bottom, left:right]
                    temp_scan_slice_4 = temp_scan_slice_4_[top:bottom, left:right]

                    temp_scan_slice_0 = np.reshape(temp_scan_slice_0, (1, new_size, new_size))
                    temp_scan_slice_1 = np.reshape(temp_scan_slice_1, (1, new_size, new_size))
                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))
                    temp_scan_slice_3 = np.reshape(temp_scan_slice_3, (1, new_size, new_size))
                    temp_scan_slice_4 = np.reshape(temp_scan_slice_4, (1, new_size, new_size))

                    scan = np.concatenate((temp_scan_slice_0, temp_scan_slice_1, temp_scan_slice_2, temp_scan_slice_3, temp_scan_slice_4), axis=0)

                elif input_dim == 3:

                    temp_scan_slice_1 = temp_scan_slice_1_[top:bottom, left:right]
                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]
                    temp_scan_slice_3 = temp_scan_slice_3_[top:bottom, left:right]

                    temp_scan_slice_1 = np.reshape(temp_scan_slice_1, (1, new_size, new_size))
                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))
                    temp_scan_slice_3 = np.reshape(temp_scan_slice_3, (1, new_size, new_size))

                    scan = np.concatenate((temp_scan_slice_1, temp_scan_slice_2, temp_scan_slice_3), axis=0)

                elif input_dim == 1:

                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]

                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))

                    scan = temp_scan_slice_2

                save_img_name = case_index + '_5_' + str(slice_no) + '_slice.tif'
                full_save_img_name = os.path.join(case_patches_folder, save_img_name)
                #
                label = temp_label_slice_[top:bottom, left:right]
                save_label_name = case_index + '_5_' + str(slice_no) + '_label.tif'
                full_save_label_name = os.path.join(case_labels_folder, save_label_name)

                #
                # print(label.max())
                #
                # dim, new_height, new_width = np.shape(scan)
                new_height, new_width = np.shape(label)

                # print(np.shape(scan))

                assert new_height == new_size
                assert new_width == new_size
                # if new_size != new_new_size:
                #     scan = resize(scan, (new_new_size, new_new_size))
                # scan = Image.fromarray(scan, mode='F')  # float32
                imsave(full_save_img_name, scan)
                # scan.save(full_save_img_name, "TIFF")
                new_height, new_width = np.shape(label)

                assert new_height == new_size
                assert new_width == new_size
                # if new_size != new_new_size:
                #     label = resize(label, (new_new_size, new_new_size))
                #     label.round()
                # label = Image.fromarray(label)  # float32
                imsave(full_save_label_name, label)
                # label.save(full_save_label_name, "TIFF")

        else:

            for slice_no in range(last_slice_no):

                position_dice = random.uniform(0, 1)

                if train_amount_slices > slices_amount:

                    sample_position = slice_no

                else:

                    sample_position = slice_no + first_slice_with_vessel

                if input_dim == 1:
                    temp_scan_slice_2_ = ct_scan[sample_position, :, :]
                elif input_dim == 3:
                    temp_scan_slice_1_ = ct_scan[sample_position - 1, :, :]
                    temp_scan_slice_2_ = ct_scan[sample_position, :, :]
                    temp_scan_slice_3_ = ct_scan[sample_position + 1, :, :]
                elif input_dim == 5:
                    temp_scan_slice_0_ = ct_scan[sample_position - 2, :, :]
                    temp_scan_slice_1_ = ct_scan[sample_position - 1, :, :]
                    temp_scan_slice_2_ = ct_scan[sample_position, :, :]
                    temp_scan_slice_3_ = ct_scan[sample_position + 1, :, :]
                    temp_scan_slice_4_ = ct_scan[sample_position + 2, :, :]

                temp_label_slice_ = label_scan[sample_position, :, :]

                # print(temp_label_slice_.shape)

                # height, width = temp_scan_slice_1.shape

                height, width = temp_label_slice_.shape

                # if height >= new_size and width >= new_size:
                assert height >= new_size
                assert width >= new_size
                # ======================
                # cropping from left up:
                # ======================
                if position_dice <= 0.25:

                    left = 0
                    right = new_size
                    top = 0
                    bottom = new_size

                elif position_dice <= 0.5:

                    left = width - new_size
                    right = width
                    top = 0
                    bottom = new_size

                elif position_dice <= 0.75:

                    left = 0
                    right = new_size
                    top = height - new_size
                    bottom = height

                else:

                    left = width - new_size
                    right = width
                    top = height - new_size
                    bottom = height

                if input_dim == 5:

                    temp_scan_slice_0 = temp_scan_slice_0_[top:bottom, left:right]
                    temp_scan_slice_1 = temp_scan_slice_1_[top:bottom, left:right]
                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]
                    temp_scan_slice_3 = temp_scan_slice_3_[top:bottom, left:right]
                    temp_scan_slice_4 = temp_scan_slice_4_[top:bottom, left:right]

                    temp_scan_slice_0 = np.reshape(temp_scan_slice_0, (1, new_size, new_size))
                    temp_scan_slice_1 = np.reshape(temp_scan_slice_1, (1, new_size, new_size))
                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))
                    temp_scan_slice_3 = np.reshape(temp_scan_slice_3, (1, new_size, new_size))
                    temp_scan_slice_4 = np.reshape(temp_scan_slice_4, (1, new_size, new_size))

                    scan = np.concatenate((temp_scan_slice_0, temp_scan_slice_1, temp_scan_slice_2, temp_scan_slice_3, temp_scan_slice_4), axis=0)

                elif input_dim == 3:

                    temp_scan_slice_1 = temp_scan_slice_1_[top:bottom, left:right]
                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]
                    temp_scan_slice_3 = temp_scan_slice_3_[top:bottom, left:right]

                    temp_scan_slice_1 = np.reshape(temp_scan_slice_1, (1, new_size, new_size))
                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))
                    temp_scan_slice_3 = np.reshape(temp_scan_slice_3, (1, new_size, new_size))

                    scan = np.concatenate((temp_scan_slice_1, temp_scan_slice_2, temp_scan_slice_3), axis=0)

                elif input_dim == 1:

                    temp_scan_slice_2 = temp_scan_slice_2_[top:bottom, left:right]

                    temp_scan_slice_2 = np.reshape(temp_scan_slice_2, (1, new_size, new_size))

                    scan = temp_scan_slice_2

                save_img_name = case_index + '_0_' + str(slice_no) + '_slice.tif'
                full_save_img_name = os.path.join(case_patches_folder, save_img_name)
                #
                label = temp_label_slice_[top:bottom, left:right]
                save_label_name = case_index + '_0_' + str(slice_no) + '_label.tif'
                full_save_label_name = os.path.join(case_labels_folder, save_label_name)

                #
                # print(label.max())
                #
                # dim, new_height, new_width = np.shape(scan)
                new_height, new_width = np.shape(label)

                # print(np.shape(scan))

                assert new_height == new_size
                assert new_width == new_size
                # if new_size != new_new_size:
                #     scan = resize(scan, (new_new_size, new_new_size))
                # scan = Image.fromarray(scan, mode='F')  # float32
                imsave(full_save_img_name, scan)
                # scan.save(full_save_img_name, "TIFF")
                new_height, new_width = np.shape(label)

                assert new_height == new_size
                assert new_width == new_size
                # if new_size != new_new_size:
                #     label = resize(label, (new_new_size, new_new_size))
                #     label.round()
                # label = Image.fromarray(label)  # float32
                imsave(full_save_label_name, label)
                # label.save(full_save_label_name, "TIFF")

        print('traning slices saving..')

    print('All cases done.')


# def divide_folds(allcases, new_resolution, input_dim):
#
#     source_folder = allcases + '/' + str(new_resolution)
#     all_cases = os.listdir(source_folder)
#     all_cases.sort()
#
#     # divided_groups = list(chunks(all_cases, 1))
#
#     train_sources = all_cases[0] + all_cases[1] + all_cases[2] + all_cases[3] + all_cases[4] + all_cases[5]
#
#     validate_sources = all_cases[6]
#
#     test_sources = all_cases[7] + all_cases[8] + all_cases[9]
#
#     # cross_folder = os.path.join(save_lo, 'training')
#     if input_dim == 3:
#         save_lo = allcases + '/3slices_corners'
#     elif input_dim == 1:
#         save_lo = allcases + '/1slice_corners'
#     elif input_dim == 5:
#         save_lo = allcases + '/5slices_corners'
#
#     validate_folder_patches = os.path.join(save_lo, 'validate/patches')
#     validate_folder_labels = os.path.join(save_lo, 'validate/labels')
#
#     train_folder_patches = os.path.join(save_lo, 'train/patches')
#     train_folder_labels = os.path.join(save_lo, 'train/labels')
#
#     test_folder_patches = os.path.join(save_lo, 'test/patches')
#     test_folder_labels = os.path.join(save_lo, 'test/labels')
#
#     try:
#         os.makedirs(save_lo)
#         os.makedirs(validate_folder_patches)
#         os.makedirs(validate_folder_labels)
#         os.makedirs(train_folder_patches)
#         os.makedirs(train_folder_labels)
#         os.makedirs(test_folder_patches)
#         os.makedirs(test_folder_labels)
#     except OSError as exc:
#         if exc.errno != errno.EEXIST:
#             raise
#     pass
#
#     for ff in validate_sources:
#         validate_source = os.path.join(source_folder, str(ff))
#         validate_source_patches = validate_source + '/patches'
#         validate_source_labels = validate_source + '/labels'
#         copy_tree(validate_source_patches, validate_folder_patches)
#         print('validation_patches are created.')
#         print('\n')
#         copy_tree(validate_source_labels, validate_folder_labels)
#         print('validation_labels are created.')
#         print('\n')
#
#     for fff in train_sources:
#         source = os.path.join(source_folder, str(fff))
#         source_patches = source + '/patches'
#         source_labels = source + '/labels'
#         copy_tree(source_patches, train_folder_patches)
#         print('train_patches are created.')
#         print('\n')
#         copy_tree(source_labels, train_folder_labels)
#         print('train_labels are created.')
#         print('\n')
#
#     for ffff in test_sources:
#         source = os.path.join(source_folder, str(ffff))
#         source_patches = source + '/patches'
#         source_labels = source + '/labels'
#
#         # print(source_patches)
#         # print(test_folder_patches)
#
#         copy_tree(source_patches, test_folder_patches)
#         print('test_patches are created.')
#         print('\n')
#         copy_tree(source_labels, test_folder_labels)
#         print('test_labels are created.')
#         print('\n')
#
#     # os.rmdir(source_folder)
#     shutil.rmtree(source_folder)


def main_loop(new_resolution, images, labels, save_lo, input_dim, train_amount_slices, case_no):
    prepare_data(new_size=new_resolution, scans_folder=images, labels_folder=labels, save_folder=save_lo, input_dim=input_dim, train_amount_slices=train_amount_slices, case_no=case_no)


if __name__ == '__main__':

    new_size = 176
    input_dim = 3
    # case_no_with_labels = 0

    # no_slices_with_labels = 30
    # images_folder = '/cluster/project0/CARVE2014_Segmentation/CARVE2014/ctscans/fullannotations'
    # labels_folder = '/cluster/project0/CARVE2014_Segmentation/CARVE2014/labels/fullannotations'
    # save_folder = '/cluster/project0/CARVE2014_Segmentation/CARVE2014'

    # images_folder = '/home/moucheng/projects_data/lung/public_data_sets/CARVE2014/ctscans/fullannotations'
    # labels_folder = '/home/moucheng/projects_data/lung/public_data_sets/CARVE2014/labels/fullannotations'
    # save_folder = '/home/moucheng/projects_data/lung/public_data_sets/CARVE2014/others'

    images_folder = '/home/moucheng/projects_data/Pulmonary_data/carve/ctscans/fullannotations'
    labels_folder = '/home/moucheng/projects_data/Pulmonary_data/carve/labels/fullannotations'
    save_folder = '/home/moucheng/projects_data/Pulmonary_data/carve/bmvc2021_rebuttal/'

    no_slices_with_labels = 30

    for i in range(10):
        main_loop(new_resolution=new_size, images=images_folder, labels=labels_folder, save_lo=save_folder, input_dim=input_dim, train_amount_slices=no_slices_with_labels, case_no=i)

print('End')

