#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
#SBATCH --constraint=kepler
module load miniconda
conda activate textattack-0.2.11

path=$1
target_model_name=$2
target_model_dataset=$3
batch_size=$4
attack_name=$5
dir_attacked_data=$6
dir_out=$7
exp_name=$8

python3 ${path}/compute_bert_scores.py \
        --target_model_name=$target_model_name \
        --target_model_dataset=$target_model_dataset \
        --batch_size=$batch_size \
        --attack_name=$attack_name \
        --dir_attacked_data=$dir_attacked_data \
        --dir_out=$dir_out \
        --exp_name=$exp_name
