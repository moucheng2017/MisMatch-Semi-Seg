#$ -l tmem=40G
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=5:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/MisMatchSSL/code/

~/miniconda3/envs/pytorch1.4/bin/python train_3D_mismatch.py \
--root_path '/SAN/medic/PerceptronHead/data/Task01_BrainTumour' \
--exp 'MisMatch_brain_exp10' \
--max_iterations 25000 \
--batch_size 8 \
--in_channel 4 \
--labeled_bs 2 \
--base_lr 0.001 \
--seed 1337 \
--width 8 \
--workers 8 \
--consistency 2.0
