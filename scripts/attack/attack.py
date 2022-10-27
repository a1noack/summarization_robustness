# -*- coding: utf-8 -*-
import logging
import os
import time
import random

import configargparse
import torch
import numpy as np

import textattack.shared.utils as textattack_utils

from summarization_robustness.wrappers import SummarizationDataset
from summarization_robustness.utils import utils
from summarization_robustness.utils import CSVLoggerAttack


def parse_summarization_attack_args():
    """This function adds a few additional parameters-specific to the summarization task–to the
    general set of parameters, and parses them.
    """
    cmd_opt = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)

    # the name of the experiment
    cmd_opt.add('--exp_name', type=str,
                help="""the unique name of this experiment; if not provided, the experiment name is generated from other
                config parameters""")

    # load parameters from YAML config file
    cmd_opt.add('--config_file0', is_config_file=True, help='yaml config file')

    # task parameters
    cmd_opt.add('--task_name', type=str, help='e.g. abuse, sentiment, summarization')

    # model parameters
    cmd_opt.add('--model_name', type=str, help='the name of the model to use')
    cmd_opt.add('--model_batch_size', type=int, default=32, help='the batch size of the model')

    # data parameters
    cmd_opt.add('--dataset_name', type=str, help='the name of the dataset to attack.')
    cmd_opt.add('--target_model_train_dataset', type=str, help='the dataset used to train the target model.')

    # attack parameters
    cmd_opt.add('--attack_toolchain', type=str,
                help='the toolchain that this attack is from; e.g. textattack or openattack')
    cmd_opt.add('--attack_name', type=str, help='the name of the attack to use')
    cmd_opt.add('--attack_n_samples', type=int, default=1000000000,
                help='the number of samples to attack; if zero, attack all samples')
    cmd_opt.add('--attack_max_queries', type=int, default=500, help='the maximum number of queries per attack')

    # i/o parameters
    cmd_opt.add('--dir_model', type=str, default='target_models/', help='central directory for trained models.')
    cmd_opt.add('--dir_dataset', type=str, default='data/', help='central directory for storing datasets.')
    cmd_opt.add('--dir_out', type=str, default='attacks/', help='central directory to store attacks.')

    # other parameters
    cmd_opt.add('--random_seed', type=int, default=1, help='the random seed value to use for reproducibility')
    cmd_opt.add('--save_every', type=int, default=25, help='no. samples to attack between saving.')

    # summarization config file
    cmd_opt.add('--config_file1', is_config_file=True,
                help='a second yaml config file for extra defaults for summarization attacks')

    # additional attack parameters
    cmd_opt.add('--attack_epsilon', type=float, help='the perturbation budget of the attacker')
    cmd_opt.add('--attack_n_perts', type=int, default=1, help='the number of perturbations to make')
    cmd_opt.add('--attack_target_rouge', type=float, default=0, help='for RougeAttack, the desired rouge score')
    cmd_opt.add('--attack_ground_truth', type=int,
                help="""1 if attacker should make output diverge from ground truth.
                0 if attacker should make output diverge from original output.""")
    cmd_opt.add('--dataset_split', type=str, help='the dataset split to use; e.g. train, test')

    # these parameters should not need to be set unless using a custom dataset
    cmd_opt.add('--model_dataset_name', type=str,
                help="""the set of weights to load for the model (e.g. cnn_dailymail, gigaword, xsum);
                if no value is passed for this parameter, this defaults to the value for dataset_name; this means
                that this parameter should really only be used if we are evaluating the model on a custom dataset""")

    # dataset specific parameters
    cmd_opt.add('--resume', type=int, help='1 to resume attacking from last index attacked, 0 to start over')

    # logger parameters
    cmd_opt.add('--log_freq', type=int, help='how many iterations between saving attack results to file')
    cmd_opt.add('--log_output_type', type=str,
                help='the output format of the csv file; `full` to write more data to file')

    # gpu parameters
    cmd_opt.add('--device', type=str, default='0', help='the device number to use')

    # set default values of general parameters for summarization
    cmd_opt.set_defaults(task_name='summarization',
                         dataset_split='test',
                         attack_toolchain='textattack',
                         attack_n_samples=0,
                         attack_ground_truth=False,
                         random_seed=1,
                         resume=1,
                         device=0,
                         log_freq=5,
                         log_output_type='darpa')

    return cmd_opt.parse_args()


if __name__ == '__main__':
    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # parse arguments
    args = parse_summarization_attack_args()
    try:
        args.device = int(args.device)
    except ValueError:
        pass
    logger.info(f'Args = {args}')

    # set random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # create logger to write attack data to
    csv_logger = CSVLoggerAttack(args)

    # finish setting up logger
    fh = logging.FileHandler(os.path.join(csv_logger.output_dir, 'output.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    logger.addHandler(ch)

    # determine which index of dataset to start at
    start_index = csv_logger.get_start_index()

    # load dataset
    s = time.time()
    dataset = SummarizationDataset(name=args.dataset_name,
                                   split='test',
                                   cache_dir=args.dir_dataset)
    logger.info(f'Loaded {args.dataset_name} dataset. {time.time() - s:.2f}s')

    # determine which samples should be attacked
    if args.attack_n_samples == 0:  # if set to zero, attack every sample in the dataset
        args.attack_n_samples = len(dataset)
    end_index = min(start_index + args.attack_n_samples, len(dataset))
    attack_indices = np.arange(start_index, end_index)
    # if dataset only has input and no ground truth output column, we cannot
    # try to make the output diverge from the ground truth output
    if not dataset.has_ground_truths:
        args.attack_ground_truth = 0

    # set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    textattack_utils.device = device
    logger.info(f'Device = {device}')

    # load model
    s = time.time()
    if args.model_dataset_name is None:
        model_dataset = args.dataset_name
    else:
        model_dataset = args.model_dataset_name
    model = utils.load_model(model_name=args.model_name,
                             dataset_name=model_dataset,
                             cache_dir=args.dir_model,
                             device=device)
    logger.info(f'Loaded {args.model_name}/{args.dataset_name} target model. {time.time() - s:.2f}s')

    # construct the attack
    s = time.time()
    attack = utils.build_attack(name=args.attack_name,
                                model=model,
                                max_queries=args.attack_max_queries,
                                epsilon=args.attack_epsilon,
                                dataset=args.dataset_name,
                                cache_dir=args.dir_model,
                                n_perturbations=args.attack_n_perts,
                                target_rouge=args.attack_target_rouge)
    logger.info(f'Loaded {args.attack_name} attacker. {time.time() - s:.2f}s')

    # attack the model for this dataset
    i = 0
    t0 = time.time()
    logger.info(f'Starting attack...')
    for attack_result in attack.attack_dataset(dataset,
                                               indices=attack_indices,
                                               attack_ground_truth=args.attack_ground_truth):
        t1 = time.time()
        try:
            csv_logger.log_attack_result(attack_result, attack_indices[i], t1 - t0)
        except IndexError:
            # if coloring doesn't work, try again without
            csv_logger.log_attack_result(attack_result, attack_indices[i], t1 - t0, color=False)

        i += 1

        # write to csv file
        if i % args.log_freq == 0:
            csv_logger.flush()
            logger.info(f'\t{i}/{len(attack_indices)}')

        t0 = time.time()
    # empty and destroy logger
    csv_logger.flush()
    del csv_logger

    logger.info('DONE!')
