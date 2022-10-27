path=/projects/uoml/anoack2/seq2seq/summarization/scripts/attack
n_samples=5000  # maximum number of samples to attack
mem=64  # RAM per job in GB
partition="preempt"
resume=1
n_perts=2
cnn_target_rouge=.25  # .25 for PEGASUS/CNN-DM and BART/CNN-DM
gig_target_rouge=.3  # each model's performance is not dropped by much at all with .5
xsu_target_rouge=.3  #  each model's performance is not dropped by much at all with .375


# PEGASUS
#bash ${path}/jobs/primer.sh $path 'pegasus' 1 'cnn_dailymail' 'adj_char_swap' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem "preemptv" $resume
#bash ${path}/jobs/primer.sh $path 'pegasus' 1 'cnn_dailymail' 'word_syn_swap' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem "preemptv" $resume
#bash ${path}/jobs/primer.sh $path 'pegasus' 5 'cnn_dailymail' 'sent_split' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem "preemptv" $resume
#bash ${path}/jobs/primer.sh $path 'pegasus' 5 'cnn_dailymail' 'sent_concat' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem "preemptv" $resume
#bash ${path}/jobs/primer.sh $path 'pegasus' 5 'cnn_dailymail' 'random_insert' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem "preemptv" $resume
#bash ${path}/jobs/primer.sh $path 'pegasus' 5 'cnn_dailymail' 'random_sent_swap' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem "preemptv" $resume
#bash ${path}/jobs/primer.sh $path 'pegasus' 1 'cnn_dailymail' 'rouge_attack' 'adams' $n_samples 10 $n_perts $cnn_target_rouge '0' $mem "preemptv" $resume

#bash ${path}/jobs/primer.sh $path 'pegasus' 5 'gigaword' 'adj_char_swap' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'pegasus' 5 'gigaword' 'word_syn_swap' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'pegasus' 5 'gigaword' 'sent_split' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'pegasus' 5 'gigaword' 'random_insert' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'pegasus' 5 'gigaword' 'rouge_attack' 'adams' $n_samples 30 $n_perts $gig_target_rouge '0' $mem $partition $resume

#bash ${path}/jobs/primer.sh $path 'pegasus' 5 'xsum' 'adj_char_swap' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'pegasus' 5 'xsum' 'word_syn_swap' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'pegasus' 5 'xsum' 'sent_split' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'pegasus' 5 'xsum' 'sent_concat' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'pegasus' 5 'xsum' 'random_insert' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'pegasus' 5 'xsum' 'random_sent_swap' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
bash ${path}/jobs/primer.sh $path 'pegasus' 5 'xsum' 'rouge_attack' 'adams' $n_samples 30 $n_perts $xsu_target_rouge '0' $mem "preemptv" $resume

## BART
#bash ${path}/jobs/primer.sh $path 'bart' 5 'cnn_dailymail' 'adj_char_swap' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'bart' 5 'cnn_dailymail' 'word_syn_swap' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'bart' 5 'cnn_dailymail' 'sent_split' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'bart' 5 'cnn_dailymail' 'sent_concat' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'bart' 5 'cnn_dailymail' 'random_insert' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'bart' 5 'cnn_dailymail' 'random_sent_swap' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'bart' 5 'cnn_dailymail' 'rouge_attack' 'adams' $n_samples 10 $n_perts $cnn_target_rouge '0' $mem $partition $resume

#bash ${path}/jobs/primer.sh $path 'bart' 5 'gigaword' 'adj_char_swap' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'bart' 5 'gigaword' 'word_syn_swap' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'bart' 5 'gigaword' 'sent_split' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'bart' 5 'gigaword' 'random_insert' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'bart' 5 'gigaword' 'rouge_attack' 'adams' $n_samples 30 $n_perts $gig_target_rouge '0' $mem $partition $resume

#bash ${path}/jobs/primer.sh $path 'bart' 5 'xsum' 'adj_char_swap' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'bart' 5 'xsum' 'word_syn_swap' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'bart' 5 'xsum' 'sent_split' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'bart' 5 'xsum' 'sent_concat' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'bart' 5 'xsum' 'random_insert' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
#bash ${path}/jobs/primer.sh $path 'bart' 5 'xsum' 'random_sent_swap' 'adams' $n_samples 10 $n_perts $target_rouge '0' $mem $partition $resume
bash ${path}/jobs/primer.sh $path 'bart' 5 'xsum' 'rouge_attack' 'adams' $n_samples 30 $n_perts $xsu_target_rouge '0' $mem "preemptv" $resume

# test run
#bash ${path}/jobs/primer.sh $path 'bart' 5 'xsum' 'rouge_attack' 'adams' $n_samples 2 $n_perts $target_rouge '0' $mem $partition $resume