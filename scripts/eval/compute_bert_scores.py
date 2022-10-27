import logging
import os
from pathlib import Path
import re
import sys

import bert_score
from bert_score.utils import (
    get_model,
    get_tokenizer,
    lang2model,
    model2layers
)
import configargparse
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
import torch

from summarization_robustness.utils import clean_str


alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]

    return sentences


def parse_eval_args():
    cmd_opt = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)

    cmd_opt.add('--target_model_name', type=str, help='the name of the model to be evaluated')
    cmd_opt.add('--target_model_dataset', type=str,
                help='the name of the dataset on which the target model was trained')
    cmd_opt.add('--batch_size', type=int, default=16, help='the batch size for the BERT model')
    cmd_opt.add('--attack_name', type=str, help='the attack name for which to get rouge results for')
    cmd_opt.add('--n', default=500, type=int, help='the number of attacks to compute rouge for')
    cmd_opt.add('--seed', default=1, type=int, help='the seed used to control sampling randomness')
    cmd_opt.add('--exp_name', default=None, type=str, help='the name of the experiment to get rouge scores for')

    # i/o parameters
    cmd_opt.add('--dir_attacked_data', type=str, help='where to load the attacked data from')
    cmd_opt.add('--dir_out', type=str, default='predictions.csv', help='where to save the output')

    return cmd_opt.parse_args()


if __name__ == '__main__':
    # create logger
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    # parse arguments for this experiment
    args = parse_eval_args()
    sys.stderr.write(f'args = {args}\n')

    # set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # load BERT model
    model_type = lang2model['en']
    num_layers = model2layers[model_type]
    tokenizer = get_tokenizer(model_type)
    model = get_model(model_type, num_layers, all_layers=False)

    # load attacked data for which to compute rouge score
    if args.exp_name != "":
        dir_data = '/projects/uoml/anoack2/seq2seq/summarization/output/attacks'
        df = pd.read_csv(os.path.join(dir_data, 'summarization', args.target_model_dataset,
                                      args.target_model_name, args.attack_name, args.exp_name, 'results.csv'))
        df['perturbed_text'] = df['perturbed_text'].str.replace('\[\[', '')
        df['perturbed_text'] = df['perturbed_text'].str.replace('\]\]', '')
    else:
        df = pd.read_csv(os.path.join(args.dir_attacked_data, 'all_attacks.csv'))
        df = df[df.model == args.target_model_name]
        df = df[df.dataset == args.target_model_dataset]
        df = df[df.attack_name == args.attack_name]

    df = df[df.result_type == 'Successful']
    df = df.sample(frac=1, random_state=args.seed)  # shuffle attacked samples in dataset
    df = df[:args.n]  # select first n samples
    logger.info(f'Computing BERT score drop for {args.n} samples.')
    logger.info(f'First 10 indices: {list(df.index[:10])}')
    print(f'Loaded and filtered dataset! Length of dataset = {len(df)}')

    # compute rouge results for all attacked samples
    all_input_similarities = []
    all_gt_to_pert_similarities = []
    all_gt_to_orig_similarities = []
    test_indices = []

    ii = 0

    for batch in np.array_split(df, -(len(df) // -args.batch_size)):  # this is equivalent to ceiling division
        print(f'Batch {ii}')
        ii += 1

        # get similarities between the INPUT texts for this batch
        original_texts = [row['original_text'] for _, row in batch.iterrows()]
        perturbed_texts = [row['perturbed_text'] for _, row in batch.iterrows()]

        input_similarites = []
        for original_text, perturbed_text in zip(original_texts, perturbed_texts):

            # split inputs into lists of sentences because the original inputs are too big for BERT
            original_text_sentences = split_into_sentences(original_text)
            perturbed_text_sentences = split_into_sentences(perturbed_text)

            # if the inputs don't have the same number of sentences
            if len(original_text_sentences) == len(perturbed_text_sentences) \
                    and 'sent' not in args.attack_name and args.target_model_dataset != 'gigaword':
                all_scores, _ = bert_score.score(
                    original_text_sentences,
                    perturbed_text_sentences,
                    model_type=None,
                    num_layers=num_layers,
                    verbose=False,
                    idf=False,
                    device=device,
                    batch_size=args.batch_size,
                    lang='en',
                    return_hash=True,
                    rescale_with_baseline=False,
                    baseline_path=None,
                )
                input_similarites.append((all_scores[2].mean().item(), all_scores[2].min().item()))  # the second position holds the F1 score

            # if the inputs don't have the same number of sentences, just do a document level bert score
            else:
                all_scores, _ = bert_score.score(
                    [original_text],
                    [perturbed_text],
                    model_type=None,
                    num_layers=num_layers,
                    verbose=False,
                    idf=False,
                    device=device,
                    batch_size=args.batch_size,
                    lang='en',
                    return_hash=True,
                    rescale_with_baseline=False,
                    baseline_path=None,
                )
                input_similarites.append((all_scores[2].item(), 0))  # the second position holds the F1 score

        # get similarities between the OUTPUT texts for this batch
        ground_truth_outputs = [row['ground_truth_output'] for _, row in batch.iterrows()]
        perturbed_outputs = [row['perturbed_output'] for _, row in batch.iterrows()]
        all_scores, _ = bert_score.score(
            ground_truth_outputs,
            perturbed_outputs,
            model_type=None,
            num_layers=num_layers,
            verbose=False,
            idf=False,
            device=device,
            batch_size=args.batch_size,
            lang='en',
            return_hash=True,
            rescale_with_baseline=False,
            baseline_path=None,
        )
        gt_to_pert_similarities = all_scores[2].tolist()  # the second position holds the F1 scores

        # get similarities between the OUTPUT texts for this batch
        original_outputs = [row['original_output'] for _, row in batch.iterrows()]
        all_scores, _ = bert_score.score(
            ground_truth_outputs,
            original_outputs,
            model_type=None,
            num_layers=num_layers,
            verbose=False,
            idf=False,
            device=device,
            batch_size=args.batch_size,
            lang='en',
            return_hash=True,
            rescale_with_baseline=False,
            baseline_path=None,
        )
        gt_to_orig_similarities = all_scores[2].tolist()  # the second position holds the F1 scores

        all_input_similarities.extend(input_similarites)
        all_gt_to_orig_similarities.extend(gt_to_orig_similarities)
        all_gt_to_pert_similarities.extend(gt_to_pert_similarities)
        test_indices.extend(batch['dataset_index'].tolist())

    # create and write cumulative rouge scores to output file
    out_dir = os.path.join(args.dir_out, f'{args.target_model_name}_{args.target_model_dataset}_{args.attack_name}')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # create dataframe with results and save it
    df = pd.DataFrame()
    df['input_similarity'] = all_input_similarities
    df['gt_to_pert_similarity'] = all_gt_to_pert_similarities
    df['gt_to_orig_similarity'] = all_gt_to_orig_similarities
    df['test_index'] = test_indices
    df.to_csv(os.path.join(out_dir, 'scores.csv'))

    logger.info(f'DONE.')
