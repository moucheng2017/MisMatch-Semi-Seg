import torch
# import sys
# sys.path.append("..")

# from Train import trainModels
# from Train_Stream_Z import trainModels
# from Train_Stream_MT import trainModels
from Train_Stream import trainModels
# from Train_Stream_DM import trainModels
# from Train_Only_labelled import trainModels
# from Train_Stream_FixMatch import trainModels
# from Train_Stream_Pseudo_Label import trainModels
# from Train_Stream_EnMT import trainModels

# torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ====================================

if __name__ == '__main__':

    # Training on ours:
    """
    when cross_validation is True,
    dataset_tag is that fold
    """
    slice = 5

    # for i in range(10):
    #
    #     fold_current = '3d_c' + str(i) + '_r176_s' + str(slice)
    #
    #     trainModels(data_directory='/home/moucheng/projects_data/lung/public_data_sets/',
    #                 dataset_name='CARVE2014',
    #                 dataset_tag=fold_current,
    #                 cross_validation=True,
    #                 input_dim=1,
    #                 class_no=2,
    #                 repeat=3,
    #                 train_batchsize=1,
    #                 augmentation='none',
    #                 num_epochs=50,
    #                 learning_rate=2e-5,
    #                 alpha=0.002,
    #                 width=24,
    #                 log_tag='IPMI2020',
    #                 consistency_loss='mse',
    #                 main_loss='dice',
    #                 self_addition=True,
    #                 save_all_segmentation=False,
    #                 annealing_mode='down',
    #                 annealing_threshold=0,
    #                 annealing_factor=0.8,
    #                 gamma=0.5
    #                 )


    # # Training on mean-teacher:
    # """
    # when cross_validation is True,
    # dataset_tag is that fold
    # """
    # trainModels(data_directory='/home/moucheng/projects_data/lung/public_data_sets/',
    #             dataset_name='CARVE2014',
    #             dataset_tag='3d_c0_r176_s50',
    #             cross_validation=False,
    #             input_dim=1,
    #             class_no=2,
    #             repeat=3,
    #             train_batchsize=1,
    #             augmentation='flip',
    #             num_epochs=20,
    #             learning_rate=2e-5,
    #             alpha=0.002,
    #             width=8,
    #             network='mean_teacher',
    #             log_tag='IPMI2020_baselines',
    #             consistency_loss='mse',
    #             main_loss='dice',
    #             self_addition=True,
    #             save_all_segmentation=False,
    #             annealing_mode='down',
    #             annealing_threshold=5,
    #             annealing_factor=0.5,
    #             )

    # # Training on entropy mean-teacher:
    # """
    # when cross_validation is True,
    # dataset_tag is that fold
    # """
    # trainModels(data_directory='/home/moucheng/projects_data/lung/public_data_sets/',
    #             dataset_name='CARVE2014',
    #             dataset_tag='3d_c0_r176_s50',
    #             cross_validation=True,
    #             input_dim=1,
    #             class_no=2,
    #             repeat=3,
    #             train_batchsize=1,
    #             augmentation='flip',
    #             num_epochs=20,
    #             learning_rate=2e-5,
    #             alpha=0.002,
    #             width=8,
    #             network='entropyMeanTeacher',
    #             log_tag='IPMI2020_baselines',
    #             consistency_loss='mse',
    #             main_loss='dice',
    #             self_addition=True,
    #             save_all_segmentation=False,
    #             annealing_mode='down',
    #             annealing_threshold=5,
    #             annealing_factor=0.5,
    #             )

    # # Training on fix-match:
    # """
    # when cross_validation is True,
    # dataset_tag is that fold
    # """
    # trainModels(data_directory='/home/moucheng/projects_data/lung/public_data_sets/',
    #             dataset_name='CARVE2014',
    #             dataset_tag='3d_c0_r176_s50',
    #             cross_validation=True,
    #             input_dim=1,
    #             class_no=2,
    #             repeat=3,
    #             train_batchsize=1,
    #             augmentation=True,
    #             num_epochs=20,
    #             learning_rate=2e-5,
    #             alpha=0.002,
    #             width=8,
    #             network='fix_match',
    #             log_tag='IPMI2020_baselines',
    #             consistency_loss='mse',
    #             main_loss='dice',
    #             self_addition=True,
    #             save_all_segmentation=False,
    #             annealing_mode='down',
    #             annealing_threshold=5,
    #             annealing_factor=0.5,
    #             )

    # # Training on pseudo-label:
    # """
    # when cross_validation is True,
    # dataset_tag is that fold
    # """
    # trainModels(data_directory='/home/moucheng/projects_data/lung/public_data_sets/',
    #             dataset_name='CARVE2014',
    #             dataset_tag='3d_c0_r176_s50',
    #             cross_validation=True,
    #             input_dim=1,
    #             class_no=2,
    #             repeat=3,
    #             train_batchsize=1,
    #             augmentation='all',
    #             num_epochs=20,
    #             learning_rate=2e-5,
    #             alpha=0.002,
    #             width=8,
    #             network='pseudo_label',
    #             log_tag='IPMI2020_baselines',
    #             consistency_loss='mse',
    #             main_loss='dice',
    #             self_addition=True,
    #             save_all_segmentation=False,
    #             annealing_mode='down',
    #             annealing_threshold=5,
    #             annealing_factor=0.5,
    #             )

    # Training on baselines:
    """
    when cross_validation is True,
    dataset_tag is that fold
    """
    trainModels(data_directory='/home/moucheng/projects_data/Brain_data/',
                dataset_name='brats2018',
                dataset_tag='t5_s20_l1_u30_WT',
                cross_validation=False,
                input_dim=4,
                class_no=2,
                repeat=3,
                train_batchsize=1,
                augmentation='none',
                num_epochs=2,
                learning_rate=2e-5,
                alpha=0.002,
                width=8,
                network='MTASSUnet',
                log_tag='IPMI2020_baselines',
                consistency_loss='mse',
                main_loss='dice',
                self_addition=True,
                save_all_segmentation=False,
                annealing_mode='down',
                annealing_threshold=5,
                annealing_factor=0.5,
                )

    # Training on only labelled and oracle:
    # """
    # when cross_validation is True,
    # dataset_tag is that fold
    # """
    # trainModels(data_directory='/home/moucheng/projects_data/lung/public_data_sets/',
    #             dataset_name='CARVE2014',
    #             dataset_tag='3d_c0_r176_s20',
    #             cross_validation=True,
    #             input_dim=1,
    #             class_no=2,
    #             repeat=3,
    #             train_batchsize=1,
    #             augmentation='all',
    #             num_epochs=50,
    #             learning_rate=2e-5,
    #             width=24,
    #             network='Unet',
    #             log_tag='IPMI2020_baselines',
    #             main_loss='dice',
    #             self_addition=True,
    #             save_all_segmentation=False
    #             )