import torch
import sys
sys.path.append("..")
from Train_Only_labelled import trainModels
# torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ====================================

if __name__ == '__main__':
    
    # Only Supervised

    slice = 30

    for i in range(10):

        fold_current = 'very_dense_' + str(i) + '_r176_s' + str(slice)

        trainModels(data_directory='/SAN/medic/PerceptronHead/data/lung/public/',
                    dataset_name='CARVE2014',
                    dataset_tag=fold_current,
                    cross_validation=True,
                    input_dim=1,
                    class_no=2,
                    repeat=1,
                    train_batchsize=1,
                    augmentation='all',
                    num_epochs=40,
                    learning_rate=2e-5,
                    width=24,
                    network='Unet',
                    log_tag='IPMI2020',
                    main_loss='dice',
                    self_addition=True,
                    save_all_segmentation=False
                    )
