path=$1
target_model_name=$2
target_model_dataset=$3
target_model_batch_size=$4

dir_target_model=/projects/uoml/anoack2/.cache
dir_dataset=/projects/uoml/anoack2/.cache
dir_out=/projects/uoml/anoack2/seq2seq/summarization/output/rouge_scores

sbatch --mem=50G \
       --time=1440 \
       --partition=gpu \
       --gres=gpu:1 \
       --job-name=${target_model_name}_${target_model_dataset}_rouge \
       --output=${path}/jobs/logs/${target_model_name}_${target_model_dataset}.out \
       --error=${path}/jobs/logs/${target_model_name}_${target_model_dataset}.err \
       ${path}/jobs/runner.sh $path $target_model_name $target_model_dataset \
       $target_model_batch_size $dir_target_model $dir_dataset $dir_out
