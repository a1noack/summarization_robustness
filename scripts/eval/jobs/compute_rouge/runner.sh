#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate textattack-0.2.11

path=$1
target_model_name=$2
target_model_dataset=$3
attack_name=$4
dir_attacked_data=$5
dir_out=$6
exp_name=$7

python3 ${path}/compute_rouge.py \
        --target_model_name=$target_model_name \
        --target_model_dataset=$target_model_dataset \
        --attack_name=$attack_name \
        --dir_attacked_data=$dir_attacked_data \
        --dir_out=$dir_out \
        --exp_name=$exp_name
