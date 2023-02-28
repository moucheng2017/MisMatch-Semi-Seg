import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if image.shape[-1] <= self.output_size[-1] or image.shape[-2] <= self.output_size[-2] or image.shape[-3] <= self.output_size[-3]:
            pw = max((self.output_size[-3] - label.shape[-3]) // 2 + 3, 0)
            ph = max((self.output_size[-2] - label.shape[-2]) // 2 + 3, 0)
            pd = max((self.output_size[-1] - label.shape[-1]) // 2 + 3, 0)

            if len(image.shape) == 3:
                image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            else:
                image = np.pad(image, [(0, 0), (pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            if label is None:
                pass
            else:
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        if len(image.shape) == 3:
            (w, h, d) = image.shape
        else:
            (c, w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[-3])
        h1 = np.random.randint(0, h - self.output_size[-2])
        d1 = np.random.randint(0, d - self.output_size[-1])

        if label is None:
            pass
        else:
            label = label[w1:w1 + self.output_size[-3], h1:h1 + self.output_size[-2], d1:d1 + self.output_size[-1]]

        if len(image.shape) == 3:
            image = image[w1:w1 + self.output_size[-3], h1:h1 + self.output_size[-2], d1:d1 + self.output_size[-1]]
        else:
            image = image[:, w1:w1 + self.output_size[-3], h1:h1 + self.output_size[-2], d1:d1 + self.output_size[-1]]

        image = (image - np.nanmean(image) + 1e-10) / (np.nanstd(image) + 1e-10)

        return {'image': image, 'label': label}


class MedicalData3D(Dataset):
    def __init__(self,
                 images_folder,
                 labels_folder=None,
                 output_shape=(96, 96, 96),
                 crop_aug=True
                 ):

        self.crop_aug_flag = crop_aug
        self.imgs_folder = images_folder
        self.lbls_folder = labels_folder

        if self.crop_aug_flag == 1:
            self.augmentation_crop = RandomCrop(output_shape)

    def __getitem__(self, index):

        all_images = sorted(glob(os.path.join(self.imgs_folder, '*.nii.*')))
        imagename = all_images[index]
        image = nib.load(imagename)
        image = image.get_fdata()
        image = np.array(image, dtype='float32')

        if len(image.shape) == 4:
            image = np.transpose(image, (3, 2, 0, 1))
        elif len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))

        if self.lbls_folder:
            all_labels = sorted(glob(os.path.join(self.lbls_folder, '*.nii.gz*')))
            label = nib.load(all_labels[index])
            label = label.get_fdata()
            label = np.array(label, dtype='float32')

            if len(label.shape) == 4:
                label = np.transpose(label, (0, 3, 1, 2))
            elif len(label.shape) == 3:
                label = np.transpose(label, (2, 0, 1))

            label[label > 0] = 1 # put all labels into one binary segmentation
            image_label = {'image': image, 'label': label}
        else:
            image_label = {'image': image, 'label': None}

        if self.crop_aug_flag == 1:
            image_label = self.augmentation_crop(image_label)

        if self.lbls_folder:

            if len(image.shape) == 3:
                image = torch.from_numpy(np.expand_dims(image_label['image'], axis=0))
            else:
                image = torch.from_numpy(image_label['image'])

            label = torch.from_numpy(image_label['label'])
            image_label = {'image': image, 'label': label}
        else:

            if len(image.shape) == 3:
                image = torch.from_numpy(np.expand_dims(image_label['image'], axis=0))
            else:
                image = torch.from_numpy(image_label['image'])

            image_label = {'image': image}

        return image_label, Path(imagename).stem

    def __len__(self):
        return len(glob(os.path.join(self.imgs_folder, '*.nii.gz')))