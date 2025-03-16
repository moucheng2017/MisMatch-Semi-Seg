## Summary
This repository contains an implementation of the MIDL2022 paper: '[Learning Morphological Feature Perturbations for Calibrated Semi-Supervised Segmentation](https://openreview.net/pdf?id=OL6tAasXCmi)' and the IEEE TMI paper '[MisMatch: Calibrated Segmentation via
Consistency on Differential Morphological
Feature Perturbations With
Limited Labels](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10121397)'. This code base was written by [Moucheng Xu](https://moucheng2017.github.io/)

## Introduction
Consistency regularisation with input data perturbations in semi-supervised classification works 
because of the cluster assumption. However, the cluster assumption does not hold in the data space in 
segmentation (https://arxiv.org/abs/1906.01916). 
Fortunately, the cluster assumption can be observed in the feature space for segmentation. 
We propose MisMatch, a novel consistency-driven semi-supervised segmentation framework which produces predictions that are invariant to learnt feature perturbations. 
MisMatch consists of an encoder and a two-head decoders. 
One decoder learns positive attention to the foreground regions of interest (RoI) on unlabelled images thereby generating dilated features. 
The other decoder learns negative attention to the foreground on the same unlabelled images thereby generating eroded features. We then apply a consistency regularisation on the paired predictions. 
MisMatch outperforms state-of-the-art semi-supervised methods on a CT-based pulmonary vessel segmentation task and a MRI-based brain tumour segmentation task. 
In addition, we show that the effectiveness of MisMatch comes from better model calibration than its supervised learning counterpart.

## Our Contributions and Method:
1) We provide a new interperation of ERF (effective receptive field: https://arxiv.org/abs/1701.04128) as a theoretical foundation for incorporating differential morphological operations of features in neural networks;
2) Based on our insight on the connection between ERF and morphological operations, we build a new encoder-decoder network architecture of semi-supervised segmentation with two decoders:
   1) positive attention decoder which enforces inductive bias to do differential dilation operations on the features;
   2) negative attention decoder which enforces another inductive bias to do differential erosion operations on the features.
3) We then apply l1 normalisation along batch dimension on the two outputs which come from the dilaiton decoder and the erosion decoder respectively before we apply a consistency loss.
![MisMatch Model.](pics/mismatch.png "Plot.")
See our paper (https://arxiv.org/pdf/2110.12179.pdf) for more details.

## Hyper-Parameters of experiments on the LA dataset
| LR   | Batch | Seed | Width | Consistency | Labels | Steps | 
|------|-------|------|-------|-------------|--------|-------|
| 0.01 | 4     | 1337 | 8     |       1     |      2 |  5000 |


## Results on the LA dataset between consistency on feature perturbations (Ours) and consistency on data perturbations (UA-MT)
| Models (5000 steps) | Dice (⬆) | Jaccard (⬆) | Hausdorff Dist. (⬇) | Average Surface Dist. (⬇) |
|:-------------------:|----------|-------------|---------------------|---------------------------|
|  MisMatch (Ours)    | 0.73     | 0.58        | 32                  | 10                        | 
|   UA-MT             | 0.70     | 0.55        | 37                  | 12                        | 

![Results on LA-Heart with different metrics.](pics/la_heart.png "Plot.")


## Installation and Usage

This repository is based on PyTorch 1.4. To use this code, please first clone the repo and install the anaconda environments via:

   ```shell

   git clone https://github.com/moucheng2017/Morphological_Feature_Perturbation_SSL

   cd MisMatchSSL

   conda env create -f midl.yml

   ```

To train the baseline on LA with default hyperparameters:, use:

   ```shell
   cd MisMatchSSL/code # change directory to your working directory where you downloaded the github repo

   python train_LA_meanteacher_certainty_unlabel.py 
   ```


To train our proposed model MisMatch on LA with default hyperparameters, use:

   ```shell
   cd MisMatchSSL/code # change directory to your working directory where you downloaded the github repo

   python train_LA_mismatch.py 
   ```

To train the models on other custom datasets or the lung tumour or the brain tumour, you have to first prepare your datasets following:
```shell
--labelled/
    -- imgs/
        -- some_case_1.nii.gz
        -- some_case_2.nii.gz
        -- ...
    -- lbls/
        -- some_case_1.nii.gz
        -- some_case_2.nii.gz
        -- ...
--unlabelled/
    -- imgs/
        -- some_case_5.nii.gz
        -- some_case_6.nii.gz
        -- ...
--test/
    --imgs/
        -- some_case_7.nii.gz
        -- some_case_8.nii.gz
        -- ...
    --lbls/
        -- some_case_7.nii.gz
        -- some_case_8.nii.gz
        -- ...

```

To train our model on your own datasets, please use the following template, but do remember to change the data directory, if you set up "witdh" as 8 and cropping size at 96 x 96 x 96 (default in the training code, but in config), then a 12GB GPU should be enough:
```shell
cd MisMatchSSL/code # change directory to your working directory where you downloaded the github repo

python train_3D_mismatch.py \
--root_path '/directory/to/your/datasets/Task01_BrainTumour' \
--exp 'MisMatch_brain' \
--max_iterations 6000 \
--batch_size 4 \
--labeled_bs 2 \
--base_lr 0.001 \
--seed 1337 \
--width 8 \
--consistency 1.0
```

## Citation

If you find our paper or code useful for your research, please consider citing:

    @inproceedings{xmc2022midl,

         title={Learning Morphological Feature Perturbations for Calibrated Semi-Supervised Segmentation},

         author={Xu, Moucheng and Zhou, Yukun and Jin, Chen and deGroot, Marius and Wilson Frederick J. and Blumberg, Stefano B. and Alexander, Daniel C. and Oxtoby, Neil P. and Jacob, Joseph},

         booktitle = {International Conference on Medical Imaging with Deep Learning (MIDL)},

         year = {2022} }

    @ARTICLE{mismatch_tmi,

         title={MisMatch: Calibrated Segmentation via Consistency on Differential Morphological Feature Perturbations With Limited Labels},

         author={Xu, Moucheng and Zhou, Yukun and Jin, Chen and deGroot, Marius and Wilson Frederick J. and Blumberg, Stefano B. and Alexander, Daniel C. and Oxtoby, Neil P. and Jacob, Joseph},

         booktitle = {IEEE Transactions on Medical Imaging},

         year = {2023} }


## Note for the LA data we used:
The left atrium processed h5 dataset is in the `data` folder. You can refer the code in `code/dataloaders/la_heart_processing.py` to process your own data. If you use the LA segmentation data, please also consider citing:

      @article{xiong2020global,

         title={A Global Benchmark of Algorithms for Segmenting Late Gadolinium-Enhanced Cardiac Magnetic Resonance Imaging},

         author={Xiong, Zhaohan and Xia, Qing and Hu, Zhiqiang and Huang, Ning and Vesal, Sulaiman and Ravikumar, Nishant and Maier, Andreas and Li, Caizi and Tong, Qianqian and Si, Weixin and others},

         journal={Medical Image Analysis},

         year={2020} }


## Note for the other two 3D datasets we used:
The lung (Task_06) and brain tumour (Task_01) datasets are downloaded from the http://medicaldecathlon.com/


## Questions
Please contact 'xumoucheng28@gmail.com' for any questions. 


## Ackwnoledgement
Massive thanks to my amazing colleagues including Yukun Zhou, Jin Chen, Marius de Groot, Fred Wilson, Neil Oxtoby, Danny Alexander and Joe Jacob.
This code base is built upon a previous public code base on consistency on data space perturbations: https://github.com/yulequan/UA-MT
