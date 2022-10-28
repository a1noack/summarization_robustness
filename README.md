# Inverstigating the robustness of summarization models
With this repository, we investigate the robustness of summarization models (specifically PEGASUS and BART) when their test set samples are transformed with various attacks (directed perturbations) and transformations (undirected perturbations).

We use two commonly referenced seq2seq adversarial attacks, Seq2Sick and MORPHEUS. Because both of these attacks were originally intended for use with translation models, we propose two new attacks more suited to the summarization task, Conceal and ROUGE Attack. 

The goal of the "Conceal" attacker is to remove the word in the original generated summary with the lowest TF-IDF score by changing the input document *without* removing the target word from the input. The goal of the ROUGE Attacker is to minimize the ROUGE score between the original generated summary and the new summary without changing more than two words in the input. 

In the table below, we show how these various attacks affect the ROUGE score (each cell in the table holds the ROUGE-2 score of the model on the perturbed test set samples). We then compare these directed attacks to various undirected textual transformations.

In the table, PEG=PEGASUS, CNN=CNN-Dailymail, Gig=Gigaword.

| Transformation name   | Transformation type | Transformations allowed | PEG/CNN       | PEG/Gig  | PEG/X-Sum | BART/CNN | BART/Gig   | BART/X-Sum  |
| ------------------- | -------------------   | --------------------- | ------------- | -------- | ------- | ------------- | -------- | ----- |
| No Attack           |NA| NA                    | 21.43         | 20.52    | 24.53   | 21.07         | 18.58    | 22.36 |
| Seq2Sick            |Directed| 1+                    | 18.2          | 11.66    | 18.45   | 18.54         | 12.41    | 16.33 |
| MORPHEUS            |Directed| 1+                  | 13.61         | 15.93    | 14.86   | 13.26         | 14.8     | 13.4  |
| Conceal             |Directed| 1+                    | 16.56         | 11       | 17.61   | 17.36         | 11.6     | 16.76 |
| ROUGE Attack        |Directed| 1-2                    | 14.08         | 8.1      | 11.69   | 15.81         | 7.6      | 12.68 |
| Adj. char. swap     |Undirected| 1                     | 20.83         | 18.59    | 24.39   | 20.66         | 17.28    | 22.14 |
| Word synonym swap   |Undirected| 1                     | 20.62         | 18.96    | 24.48   | 20.24         | 17.51    | 22.19 |
| Add punctuation     |Undirected| 1                     | 20.61         | 20.3     | 24.43   | 19.4          | 18.68    | 22.28 |
| Remove punctuation  |Undirected| 1                     | 20.61         | \-       | 24.56   | 20.72         | \-       | 22.4  |
| Swap sentence order |Undirected| 1                     | 20.24         | \-       | 24.44   | 20.85         | \-       | 22.33 |
| Append irrelevant   |Undirected| 1                     | 20.54         | 13.99    | 24.51   | 18.77         | 29.15    | 22.29 |
| Adj. char. swap     |Undirected| 2                     | \-            | 17.19    | 23.61   | 20.13         | 16.17    | 21.85 |
| Word synonym swap   |Undirected| 2                     | \-            | 18.21    | 23.86   | 19.98         | 17.32    | 21.56 |
| Add punctuation     |Undirected| 2                     | 20            | 20.74    | 23.63   | 20.02         | 18.55    | 21.6  |
| Remove punctuation  |Undirected| 2                     | 20.13         | \-       | 22.13   | 20.79         | \-       | 19.98 |
| Swap sentence order |Undirected| 2                     | 20.38         | \-       | 21.12   | 19.94         | \-       | 19.06 |
| Append irrelevant   |Undirected| 2                     | 20.91         | 19.67    | 23.67   | 20.31         | 18.73    | 22.16 |
