import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm


def test_all_case(net, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, network_flag=0):
    total_metric = 0.0
    for image_path in tqdm(image_list):
        id = image_path.split('/')[-1]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, network_flag=network_flag)

        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    return avg_metric


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, network_flag=0):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                if network_flag == 0:
                    y = net(test_patch)
                    y = F.softmax(y, dim=1)
                elif network_flag == 1:
                    y1, y2 = net(test_patch)
                    y1 = F.softmax(y1, dim=1)
                    y2 = F.softmax(y2, dim=1)
                    y = (y1 + y2) / 2
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction==i)
        label_tmp = (label==i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd

#
# def preprocessing_accuracy(label_true, label_pred, n_class):
#
#     # print(type(label_pred).__module__)
#     # print(type(label_true))
#
#     if type(label_pred).__module__ == np.__name__:
#         label_pred = np.asarray(label_pred, dtype='int32')
#     else:
#         if n_class == 2:
#         # thresholding predictions:
#             output_zeros = torch.zeros_like(label_pred)
#             output_ones = torch.ones_like(label_pred)
#             label_pred = torch.where((label_pred > 0.5), output_ones, output_zeros)
#         label_pred = label_pred.cpu().detach()
#         label_pred = np.asarray(label_pred, dtype='int32')
#
#     if type(label_true).__module__ == np.__name__:
#         label_true = np.asarray(label_true, dtype='int32')
#     else:
#         label_true = label_true.cpu().detach()
#         label_true = np.asarray(label_true, dtype='int32')
#
#     return label_pred, label_true
#
# # https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
# # https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/utils/metrics.py
# # https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py
#
#
# def _fast_hist(label_true, label_pred, n_class):
#     # label_pred, label_true = preprocessing_accuracy(label_true, label_pred, n_class)
#     mask = (label_true >= 0) & (label_true < n_class)
#
#     # print(np.shape(mask))
#     # print(np.shape(label_true))
#     # print(np.shape(label_pred))
#
#     hist = np.bincount(
#         n_class * label_true[mask].astype(int) +
#         label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
#     return hist
#
#
# def segmentation_scores(label_trues, label_preds, n_class):
#     """Returns accuracy score evaluation result.
#       - overall accuracy
#       - mean accuracy
#       - mean IU
#       - fwavacc
#     """
#     label_preds, label_trues = preprocessing_accuracy(label_trues, label_preds, n_class)
#     hist = np.zeros((n_class, n_class))
#     for lt, lp in zip(label_trues, label_preds):
#         hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
#     # acc = np.diag(hist).sum() / hist.sum()
#     # acc_cls = np.diag(hist) / hist.sum(axis=1)
#     # acc_cls = np.nanmean(acc_cls)
#     # iou:
#     iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
#     mean_iou = np.nanmean(iu)
#
#     # freq = hist.sum(axis=1) / hist.sum()
#     # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#     # iflat = label_preds.view(-1)
#     # tflat = label_trues.view(-1)
#     # intersection = (iflat * tflat).sum()
#     # union = iflat.sum() + tflat.sum()
#
#     return mean_iou