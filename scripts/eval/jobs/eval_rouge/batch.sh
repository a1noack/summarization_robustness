# pegasus
path=/projects/uoml/anoack2/seq2seq/summarization/scripts/eval
bash ${path}/jobs/primer.sh $path pegasus cnn_dailymail 10
bash ${path}/jobs/primer.sh $path pegasus gigaword 100
bash ${path}/jobs/primer.sh $path pegasus xsum 20
# bart
bash ${path}/jobs/primer.sh $path bart cnn_dailymail 10
bash ${path}/jobs/primer.sh $path bart gigaword 100
bash ${path}/jobs/primer.sh $path bart xsum 20
