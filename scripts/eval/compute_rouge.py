import configargparse
import logging
import os
import pandas as pd
from pathlib import Path
from rouge_score import rouge_scorer
from summarization_robustness.utils import clean_str


def parse_eval_args():
    cmd_opt = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)

    cmd_opt.add('--target_model_name', type=str, help='the name of the model to be evaluated')
    cmd_opt.add('--target_model_dataset', type=str,
                help='the name of the dataset on which the target model was trained')
    cmd_opt.add('--attack_name', type=str, help='the attack name for which to get rouge results for')
    cmd_opt.add('--n', default=1000000, type=int, help='the number of attacks to compute rouge for')
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

    # create rouge scorer object
    score_types = ['rouge1', 'rouge2', 'rougeLsum']
    scorer = rouge_scorer.RougeScorer(score_types, use_stemmer=True)

    # load attacked data for which to compute rouge score
    if args.exp_name != "":
        df = pd.read_csv(os.path.join(args.dir_attacked_data, 'summarization', args.target_model_dataset,
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
    logger.info(f'Computing ROUGE scores for first {args.n} randomly selected samples.')
    logger.info(f'First 10 indices: {list(df.index[:10])}')

    # create dictionary object for holding rouge results
    results = dict.fromkeys(score_types, 0)

    # compute rouge results for all attacked samples
    for idx, row in df.iterrows():
        ground_truth_output = clean_str(row['ground_truth_output'])
        perturbed_output = clean_str(row['perturbed_output'])
        scores = scorer.score(ground_truth_output, perturbed_output)

        # add rouge results for this sample to the accumulator
        for score_type in score_types:
            results[score_type] += scores[score_type].fmeasure

    # create and write cumulative rouge scores to output file
    Path(args.dir_out).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(args.dir_out,
                               f'{args.target_model_name}_{args.target_model_dataset}_{args.attack_name}.txt')
    with open(output_file, 'a') as f:
        logger.info(f'ROUGE-1 / ROUGE-2 / ROUGE-L')
        f.write('ROUGE-1 / ROUGE-2 / ROUGE-L\n')
        f.write(f'{args.n} random samples; seed = {args.seed}\n')
        output_string = ''
        for score_type, score in results.items():
            avg = score / len(df)
            output_string += f'{avg * 100:.2f}'
            if score_type != 'rougeLsum':
                output_string += ' / '
        logger.info(output_string)
        f.write(f'{output_string}\n')
        logger.info(f'Successfully wrote average ROUGE scores to file.')
