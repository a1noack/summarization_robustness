#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
#SBATCH --constraint=kepler  # this is to prevent the A100s from being used -- use K80s instead
module load miniconda
conda activate textattack-0.2.11

exp_name='test_attacks4'
model_name=$1
model_batch_size=$2
dataset_name=$3
attack_name=$4
attack_toolchain=$5
attack_n_samples=$6
attack_max_queries=$7
attack_n_perts=$8
attack_target_rouge=$9
device=${10}
resume=${11}
dir_model='/projects/uoml/anoack2/.cache'
dir_dataset='/projects/uoml/anoack2/.cache'
dir_out='/projects/uoml/anoack2/seq2seq/summarization/output'
log_output_type='full'


python scripts/attack/attack.py \
          --exp_name=$exp_name \
          --model_name=$model_name \
          --model_batch_size=$model_batch_size \
          --dataset_name=$dataset_name \
          --attack_toolchain=$attack_toolchain \
          --attack_name=$attack_name \
          --attack_n_samples=$attack_n_samples \
          --attack_max_queries=$attack_max_queries \
          --attack_target_rouge=$attack_target_rouge \
          --dir_model=$dir_model \
          --dir_dataset=$dir_dataset \
          --dir_out=$dir_out \
          --log_output_type=$log_output_type \
          --device=$device \
          --attack_n_perts=$attack_n_perts \
          --resume=$resume
