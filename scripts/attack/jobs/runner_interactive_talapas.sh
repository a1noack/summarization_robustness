#!/bin/bash

exp_name='test_attacks2'
model_name='pegasus'
model_batch_size=2
dataset_name='gigaword'
attack_toolchain='textattack'
attack_name='rouge_attack'
attack_n_samples=2
attack_max_queries=5
attack_target_rouge=.17
dir_model='/projects/uoml/anoack2/.cache'
dir_dataset='/projects/uoml/anoack2/.cache'
dir_out='output'
log_output_type='full'
device=0
resume=0
n_perts=2

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
          --device=$device \
          --resume=$resume \
          --attack_n_perts=$n_perts \
          --attack_target_rouge=$attack_target_rouge
