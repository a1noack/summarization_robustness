#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate textattack-0.2.11

path=$1
target_model_name=$2
target_model_dataset=$3
target_model_batch_size=$4
dir_target_model=$5
dir_dataset=$6
dir_out=$7

python3 ${path}/eval_rouge.py \
        --target_model_name=$target_model_name \
        --target_model_dataset=$target_model_dataset \
        --target_model_batch_size=$target_model_batch_size \
        --dir_target_model=$dir_target_model \
        --dir_dataset=$dir_dataset \
        --dir_out=$dir_out

