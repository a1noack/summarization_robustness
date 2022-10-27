import configargparse
import logging
import os
from pathlib import Path
import torch
from rouge_score import rouge_scorer
from summarization_robustness.utils import load_model, clean_str
from summarization_robustness.wrappers import SummarizationDataset


def parse_eval_args():
    cmd_opt = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)

    cmd_opt.add('--target_model_name', type=str, help='the name of the model to be evaluated')
    cmd_opt.add('--target_model_batch_size', type=int, help='the batch size for this model/dataset')
    cmd_opt.add('--target_model_dataset', type=str,
                help='the name of the dataset on which the target model was trained')

    # i/o parameters
    cmd_opt.add('--dir_target_model', type=str, help='where to load the target model weights from')
    cmd_opt.add('--dir_dataset', type=str,
                help='where the csv file with the possibly perturbed samples to be classified can be found')
    cmd_opt.add('--dir_out', type=str, default='predictions.csv',
                help='where to save the output for this classification experiment')

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
    batch_size = args.target_model_batch_size

    # instantiate rougeScorer object
    score_types = ['rouge1', 'rouge2', 'rougeLsum']
    scorer = rouge_scorer.RougeScorer(score_types, use_stemmer=True)

    # set device
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device = {device}.')

    # load model
    model = load_model(args.target_model_name, args.target_model_dataset, args.dir_target_model, device)
    logger.info(f'Loaded {args.target_model_name} model trained on {args.target_model_dataset}.')

    # load test split of dataset
    dataset = SummarizationDataset(name=args.target_model_dataset,
                                   split='test',
                                   cache_dir=args.dir_dataset)
    logger.info(f'Loaded {args.target_model_dataset} dataset.')

    # create evaluation accumulators
    rouge_scores = dict.fromkeys(score_types, 0)

    # evaluate model on test set
    logger.info(f'Beginning ROUGE evaluation.')
    percent_done = .1
    for i in range(0, len(dataset), batch_size):
        j = min(i + batch_size, len(dataset))
        batch_idxs = list(range(i, j))  # stop attempt to get idxs too large
        try:
            input_text_list, ground_truth_outputs = list(zip(*dataset[batch_idxs]))
        except IndexError:
            logger.info(f'batch_idxs = {batch_idxs}')
        predicted_outputs = model(input_text_list)

        # for each ground truth output / predicted output pair, calculate rouge scores
        for ground_truth, predicted in zip(ground_truth_outputs, predicted_outputs):
            ground_truth, predicted = clean_str(ground_truth), clean_str(predicted)
            scores = scorer.score(ground_truth, predicted)

            # update the cumulative results object
            for score_type in score_types:
                rouge_scores[score_type] += scores[score_type].fmeasure

        # log cumulative rouge scores
        if j / len(dataset) >= percent_done:
            logger.info(f'Cumulative ROUGE after {percent_done*100:.1f}% of samples:')
            for score_type, score in rouge_scores.items():
                avg = score / j  # account for overshooting of final batch
                logger.info(f'\t{score_type}: {avg*100:.2f}')
            percent_done += .1

    # create and write cumulative rouge scores to output file
    Path(args.dir_out).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(args.dir_out, f'{args.target_model_name}_{args.target_model_dataset}.txt')
    with open(output_file, 'w') as f:
        logger.info(f'Writing average ROUGE scores to file.')
        f.write('ROUGE-1 / ROUGE-2 / ROUGE-L\n')
        for score_type, score in rouge_scores.items():
            avg = score / len(dataset)
            f.write(f'{avg*100:.2f}')
            if score_type == 'rougeLsum':
                f.write('\n')
            else:
                f.write(' / ')
