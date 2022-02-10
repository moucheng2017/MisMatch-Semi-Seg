import torch
import numpy as np
# import sys
# sys.path.append("..")
from Train_Stream_U import trainModels
# torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ====================================

if __name__ == '__main__':

    # Ablation studies on the decoder:
    """
    when cross_validation is True,
    dataset_tag is that fold
    """
    slice = 5

    for i in np.arange(0, 10, 1):

        fold_current = 'dense_' + str(i) + '_r176_s' + str(slice)

        trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data/',
                    dataset_name='carve',
                    dataset_tag=fold_current,
                    cross_validation=True,
                    input_dim=1,
                    class_no=2,
                    repeat=1,
                    train_batchsize=1,
                    augmentation='none',
                    num_epochs=50,
                    learning_rate=2e-5,
                    alpha=0.002,
                    width=24,
                    log_tag='BMVC2021_Rebuttal2',
                    consistency_loss='mse',
                    main_loss='dice',
                    save_all_segmentation=False,
                    self_loss_type='jacobi_all',
                    annealing_mode='down',
                    annealing_threshold=0,
                    annealing_factor=0.5,
                    gamma=0.2
                    )

