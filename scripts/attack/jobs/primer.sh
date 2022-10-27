path=$1
model_name=$2
model_batch_size=$3
dataset_name=$4
attack_name=$5
attack_toolchain=$6
attack_n_samples=$7
attack_max_queries=$8
attack_n_perts=$9
target_rouge=${10}
device=${11}
mem=${12}
partition=${13}
resume=${14}


if [[ "$partition" == "gpu" ]]
then
  sbatch --mem=${mem}G \
         --time=1440 \
         --partition=gpu \
         --gres=gpu:1 \
         --job-name=A_${model_name}_${dataset_name}_${attack_name} \
         --output=${path}/jobs/logs/A_${model_name}_${dataset_name}_${attack_name}.out \
         --error=${path}/jobs/errors/A_${model_name}_${dataset_name}_${attack_name}.err \
         ${path}/jobs/runner.sh $model_name $model_batch_size $dataset_name \
         $attack_name $attack_toolchain $attack_n_samples $attack_max_queries \
         $attack_n_perts $target_rouge $device $resume
elif [[ "$partition" == "preempt" ]]
then
  sbatch --mem=${mem}G \
         --time=1440 \
         --partition=preempt \
         --requeue \
         --gres=gpu:1 \
         --job-name=A_${model_name}_${dataset_name}_${attack_name} \
         --output=${path}/jobs/logs/A_${model_name}_${dataset_name}_${attack_name}.out \
         --error=${path}/jobs/errors/A_${model_name}_${dataset_name}_${attack_name}.err \
         ${path}/jobs/runner.sh $model_name $model_batch_size $dataset_name \
         $attack_name $attack_toolchain $attack_n_samples $attack_max_queries \
         $attack_n_perts $target_rouge $device $resume
elif [[ "$partition" == "preemptv" ]]
then
  sbatch --mem=${mem}G \
         --time=1440 \
         --partition=preempt \
         --constraint="volta" \
         --requeue \
         --gres=gpu:1 \
         --job-name=A_${model_name}_${dataset_name}_${attack_name} \
         --output=${path}/jobs/logs/A_${model_name}_${dataset_name}_${attack_name}.out \
         --error=${path}/jobs/errors/A_${model_name}_${dataset_name}_${attack_name}.err \
         ${path}/jobs/runner.sh $model_name $model_batch_size $dataset_name \
         $attack_name $attack_toolchain $attack_n_samples $attack_max_queries \
         $attack_n_perts $target_rouge $device $resume
else
  sbatch --mem=${mem}G \
         --time=1440 \
         --partition=$partition \
         --gres=gpu:1 \
         --job-name=A_${model_name}_${dataset_name}_${attack_name} \
         --output=${path}/jobs/logs/A_${model_name}_${dataset_name}_${attack_name}.out \
         --error=${path}/jobs/errors/A_${model_name}_${dataset_name}_${attack_name}.err \
         ${path}/jobs/runner.sh $model_name $model_batch_size $dataset_name \
         $attack_name $attack_toolchain $attack_n_samples $attack_max_queries \
         $attack_n_perts $target_rouge $device $resume
fi
