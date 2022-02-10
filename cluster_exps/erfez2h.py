import torch
import sys
sys.path.append("..")
from Train_Stream_Z import trainModels
# torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ====================================

if __name__ == '__main__':
    
    # Ours

    slice = 5

    for i in range(10):

        fold_current = '3d_c' + str(i) + '_r176_s' + str(slice)

        trainModels(data_directory='/SAN/medic/PerceptronHead/data/lung/public/',
                    dataset_name='CARVE2014',
                    dataset_tag=fold_current,
                    cross_validation=True,
                    input_dim=1,
                    class_no=2,
                    repeat=1,
                    train_batchsize=1,
                    augmentation='none',
                    num_epochs=50,
                    learning_rate=2e-5,
                    alpha=0.004,
                    width=24,
                    log_tag='IPMI2020',
                    consistency_loss='mse',
                    main_loss='dice',
                    self_addition=True,
                    save_all_segmentation=False,
                    annealing_mode='down',
                    annealing_threshold=0,
                    annealing_factor=0.8,
                    self_loss_type='jacobi_all',
                    gamma=0.2
                    )

