path=/projects/uoml/anoack2/seq2seq/summarization/scripts/eval
target_model_name=pegasus
target_model_dataset=xsum
batch_size=2
attack_name=seq2sick
dir_attacked_data=/projects/uoml/anoack2/react/summarization/attacked_data/attacks/summarization
dir_out=/projects/uoml/anoack2/seq2seq/summarization/output/bert_scores
exp_name=""

bash ${path}/jobs/compute_bert_scores/runner.sh $path $target_model_name $target_model_dataset \
      $batch_size $attack_name $dir_attacked_data $dir_out $exp_name