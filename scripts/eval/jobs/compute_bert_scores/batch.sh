# transformations
path=/projects/uoml/anoack2/seq2seq/summarization/scripts/eval
batch_size=2
attack_names=("adj_char_swap" "word_syn_swap" "sent_split" "sent_concat" "random_insert" "random_sent_swap")
exp_name="test_attacks2"

for attack_name in "${attack_names[@]}"
do
  # pegasus
  bash ${path}/jobs/compute_bert_scores/primer.sh $path pegasus cnn_dailymail $batch_size $attack_name $exp_name
  bash ${path}/jobs/compute_bert_scores/primer.sh $path pegasus gigaword $batch_size $attack_name $exp_name
  bash ${path}/jobs/compute_bert_scores/primer.sh $path pegasus xsum $batch_size $attack_name $exp_name
  # bart
  bash ${path}/jobs/compute_bert_scores/primer.sh $path bart cnn_dailymail $batch_size $attack_name $exp_name
  bash ${path}/jobs/compute_bert_scores/primer.sh $path bart gigaword $batch_size $attack_name $exp_name
  bash ${path}/jobs/compute_bert_scores/primer.sh $path bart xsum $batch_size $attack_name $exp_name
done

bash ${path}/jobs/compute_bert_scores/primer.sh $path pegasus cnn_dailymail $batch_size "word_syn_swap" "test_attacks"

# OG attacks
attack_names=("seq2sick" "morpheus" "conceal")
exp_name=""

for attack_name in "${attack_names[@]}"
do
  # pegasus
  bash ${path}/jobs/compute_bert_scores/primer.sh $path pegasus cnn_dailymail $batch_size $attack_name $exp_name
  bash ${path}/jobs/compute_bert_scores/primer.sh $path pegasus gigaword $batch_size $attack_name $exp_name
  bash ${path}/jobs/compute_bert_scores/primer.sh $path pegasus xsum $batch_size $attack_name $exp_name
  # bart
  bash ${path}/jobs/compute_bert_scores/primer.sh $path bart cnn_dailymail $batch_size $attack_name $exp_name
  bash ${path}/jobs/compute_bert_scores/primer.sh $path bart gigaword $batch_size $attack_name $exp_name
  bash ${path}/jobs/compute_bert_scores/primer.sh $path bart xsum $batch_size $attack_name $exp_name
done
