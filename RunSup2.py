import torch
# import sys
# sys.path.append("..")
# from Train import trainModels
from Train_Only_labelled import trainModels
# torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    for i in [2, 3, 4]:

        slice = 100

        fold_current = 'sparse_' + str(i) + '_r176_s' + str(slice)

        trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data/',
                    dataset_name='carve',
                    dataset_tag=fold_current,
                    cross_validation=True,
                    input_dim=1,
                    class_no=2,
                    repeat=1,
                    train_batchsize=1,
                    augmentation='none',
                    num_epochs=40,
                    learning_rate=2e-5,
                    width=24,
                    network='ERFAnet',
                    log_tag='bmvc2021',
                    main_loss='dice',
                    self_addition=True,
                    save_all_segmentation=False
                    )
