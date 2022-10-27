# pegasus
path=/projects/uoml/anoack2/seq2seq/summarization/scripts/eval
#attack_names=("adj_char_swap" "word_syn_swap" "sent_split" "sent_concat" "random_insert" "random_sent_swap")
attack_names=("rouge_attack")
exp_name="test_attacks3"

for attack_name in "${attack_names[@]}"
do
  # pegasus
#  bash ${path}/jobs/compute_rouge/primer.sh $path pegasus cnn_dailymail $attack_name $exp_name
#  bash ${path}/jobs/compute_rouge/primer.sh $path pegasus gigaword $attack_name $exp_name
  bash ${path}/jobs/compute_rouge/primer.sh $path pegasus xsum $attack_name $exp_name
  # bart
#  bash ${path}/jobs/compute_rouge/primer.sh $path bart cnn_dailymail $attack_name $exp_name
#  bash ${path}/jobs/compute_rouge/primer.sh $path bart gigaword $attack_name $exp_name
  bash ${path}/jobs/compute_rouge/primer.sh $path bart xsum $attack_name $exp_name
done
