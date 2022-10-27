#!/bin/bash

exp_name='test_attacks'
model_name='pegasus'
model_batch_size=4
dataset_name='gigaword'
attack_toolchain='textattack'
attack_name='morpheus'
attack_n_samples=2
attack_max_queries=10
dir_model='/data/anoack2/.cache'
dir_dataset='/data/anoack2/.cache'
dir_out='output'
log_output_type='full'
device=0

python scripts/attack/attack.py \
          --exp_name=$exp_name \
          --model_name=$model_name \
          --model_batch_size=$model_batch_size \
          --dataset_name=$dataset_name \
          --attack_toolchain=$attack_toolchain \
          --attack_name=$attack_name \
          --attack_n_samples=$attack_n_samples \
          --attack_max_queries=$attack_max_queries \
          --dir_model=$dir_model \
          --dir_dataset=$dir_dataset \
          --dir_out=$dir_out \
          --log_output_type=$log_output_type \
          --device=$device