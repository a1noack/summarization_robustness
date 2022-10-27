path=$1
target_model_name=$2
target_model_dataset=$3
batch_size=$4
attack_name=$5
exp_name=$6

dir_attacked_data=/projects/uoml/anoack2/react/summarization/attacked_data/attacks/summarization
dir_out=/projects/uoml/anoack2/seq2seq/summarization/output/bert_scores

sbatch --mem=40G \
       --time=1440 \
       --partition=gpu \
       --job-name=${target_model_name}_${target_model_dataset}_${attack_name}_bert_scores \
       --output=${path}/jobs/logs/${target_model_name}_${target_model_dataset}_${attack_name}_bert_score.out \
       --error=${path}/jobs/logs/${target_model_name}_${target_model_dataset}_${attack_name}_bert_score.err \
       ${path}/jobs/compute_bert_scores/runner.sh $path $target_model_name $target_model_dataset \
       $batch_size $attack_name $dir_attacked_data $dir_out $exp_name
