#$ -l tmem=40G
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=24:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/MisMatchSSL/code/

~/miniconda3/envs/pytorch1.4/bin/python train_3D_mismatch.py \
--root_path '/SAN/medic/PerceptronHead/data/Task01_BrainTumour' \
--exp 'MisMatch_brain_exp2' \
--max_iterations 4000 \
--batch_size 4 \
--in_channel 4 \
--labeled_bs 2 \
--base_lr 0.0001 \
--seed 1337 \
--width 8 \
--consistency 1.0
