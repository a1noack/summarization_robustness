path=$1
target_model_name=$2
target_model_dataset=$3
attack_name=$4
exp_name=$5

dir_attacked_data=/projects/uoml/anoack2/seq2seq/summarization/output/attacks
dir_out=/projects/uoml/anoack2/seq2seq/summarization/output/rouge_scores

sbatch --mem=10G \
       --time=1440 \
       --partition=short \
       --job-name=${target_model_name}_${target_model_dataset}_${attack_name}_rouge \
       --output=${path}/jobs/logs/${target_model_name}_${target_model_dataset}_${attack_name}.out \
       --error=${path}/jobs/logs/${target_model_name}_${target_model_dataset}_${attack_name}.err \
       ${path}/jobs/compute_rouge/runner.sh $path $target_model_name $target_model_dataset \
       $attack_name $dir_attacked_data $dir_out $exp_name
