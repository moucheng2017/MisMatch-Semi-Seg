### News
[2022 Sep 15th] For easier maintenance (also as requested by the reviewers of the journal version of our paper), we release a new version of our implementation based on the data and the code from a previous well-known paper '[Uncertainty-aware Self-ensembling Model for Semi-supervised 3D Left Atrium Segmentation](https://arxiv.org/abs/1907.07034)' which was written by [Lequan Yu](https://yulequan.github.io/)

[2022 Oct 28th] We aim to clean up and rewrite our old code on CARVE and release results on a new dataset.

### Introduction
This repository is an implementation of our MIDL 2022 Oral paper '[Learning Morphological Operations for Calibrated Semi-Supervised Segmentation](https://openreview.net/pdf?id=OL6tAasXCmi)' on a public available dataset which was not included in the original MIDL paper. This code base was written and maintained by [Moucheng Xu](https://moucheng2017.github.io/)

[//]: # (### Hyper-parameters)

[//]: # (We tested the following hyper-parameters:)

[//]: # ()
[//]: # (### Results)

[//]: # ()
[//]: # ()
[//]: # (### Methods)

[//]: # ()
[//]: # ()
[//]: # (### Calibration and Consistency Driven Semi-Supervised Learning)



[//]: # (### Installation and Usage)

[//]: # (This repository is based on PyTorch 1.4. To use this code, please first clone the repo and install the anaconda environments via:)

[//]: # (   ```shell)

[//]: # (   git clone https://github.com/moucheng2017/Morphological_Feature_Perturbation_SSL)

[//]: # (   cd MisMatch)

[//]: # (   conda env create -f midl.yml)

[//]: # (   ```)

[//]: # (To train the baseline, use:)

[//]: # (   ```shell)

[//]: # (   cd code)

[//]: # (   python train_LA_meanteacher_certainty_unlabel.py --gpu 0)

[//]: # (   ```)

[//]: # ()
[//]: # (To train our proposed model MisMatch, use:)

[//]: # (   ```shell)

[//]: # (   cd code)

[//]: # (   python train_LA_meanteacher_certainty_unlabel.py --gpu 0)

[//]: # (   ```)

[//]: # (### Citation)

[//]: # ()
[//]: # (If you find our paper or code useful for your research, please consider citing:)

[//]: # ()
[//]: # (    @inproceedings{xu2022midl,)

[//]: # (         title={Learning Morphological Operations in Calibrated Semi-Supervised Segmentation},)

[//]: # (         author={Xu, Moucheng and Zhou, Yukun and},)

[//]: # (         booktitle = {MIDL},)

[//]: # (         year = {2022} })

[//]: # ()
[//]: # (If you use the LA segmentation data, please also consider citing:)

[//]: # ()
[//]: # (      @article{xiong2020global,)

[//]: # (         title={A Global Benchmark of Algorithms for Segmenting Late Gadolinium-Enhanced Cardiac Magnetic Resonance Imaging},)

[//]: # (         author={Xiong, Zhaohan and Xia, Qing and Hu, Zhiqiang and Huang, Ning and Vesal, Sulaiman and Ravikumar, Nishant and Maier, Andreas and Li, Caizi and Tong,          Qianqian and Si, Weixin and others},)

[//]: # (         journal={Medical Image Analysis},)

[//]: # (         year={2020} })

### Note for data
The left atrium dataset We provided the processed h5 data in the `data` folder. You can refer the code in `code/dataloaders/la_heart_processing.py` to process your own data.


### Questions
Please contact 'xumoucheng28@gmail.com' or 'rmapmxu@ucl.ac.uk'


[//]: # (### Ackwnoledgement)

[//]: # (Our code uses the infrascture and the data from this repo:. We thank the developers of the community who contributed to the referred repo.)
