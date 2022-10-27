from pathlib import Path
import sys
import time

import os
import pandas as pd

from textattack.loggers import CSVLogger
from textattack.shared import AttackedText
from textattack.attack_recipes import Seq2SickCheng2018BlackBox as Seq2Sick
from textattack.attack_recipes import MorpheusTan2020 as Morpheus
# from summarization.attack_recipes.morpheus_mod import MorpheusMod
# from summarization.attack_recipes.conceal import Conceal

from seq2seq.summarization.wrappers.models import PEGASUS, BART
from seq2seq.summarization.wrappers.attacks import (
    AdjacentCharSwap, WordSynonymSwap,
    SentenceSplit, SentenceConcat,
    RandomStringInsert, SentenceSwap,
    RougeAttack
)


def load_model(model_name, dataset_name, cache_dir, device):
    if model_name == 'pegasus':
        model = PEGASUS(dataset_name, cache_dir, device)
    elif model_name == 'bart':
        model = BART(dataset_name, cache_dir, device)
    else:
        raise ValueError(f'{model_name} is not a supported model!')

    return model


def build_attack(name, model, max_queries, epsilon=30, n_perturbations=1,
                 target_rouge=0, dataset='', cache_dir=''):
    '''Builds a textattack.AttackRecipe attack instance using the provided
    model.

    Args:
        name (str): The name of the attack to use. E.g. Seq2Sick or Morpheus.
        model (models.TextAttackModel): The model to use to build attack. E.g. models.Pegasus.
        max_queries (int): The maximum number of queries to the model to make per attack.
        distance (int): The maximum Levenshtein edit distance value for Seq2Sick perturbation.
        dataset (str): Used to retrieve TF-IDF vectorizer.
        cache_dir (str): Used to retrieve TF-IDF vectorizer.
    '''
    if name == 'seq2sick':
        attack = Seq2Sick.build(model, epsilon=int(epsilon))
    elif name == 'morpheus':
        attack = Morpheus.build(model)
    elif name == 'rouge_attack':
        attack = RougeAttack.build(model, target_rouge=target_rouge)
    elif name == 'adj_char_swap':
        attack = AdjacentCharSwap.build(model, n_perturbations)
    elif name == 'word_syn_swap':
        attack = WordSynonymSwap.build(model, n_perturbations=n_perturbations)
    elif name == 'sent_split':
        attack = SentenceSplit.build(model, n_perturbations)
    elif name == 'sent_concat':
        attack = SentenceConcat.build(model, n_perturbations)
    elif name == 'random_insert':
        attack = RandomStringInsert.build(model, n_perturbations)
    elif name == 'random_sent_swap':
        attack = SentenceSwap.build(model, n_perturbations)
    # elif name == 'morpheus_mod':
    #     attack = MorpheusMod.build(model)
    # elif name == 'conceal':
    #     attack = Conceal.build(model, dataset=dataset, cache_dir=cache_dir)
    else:
        raise ValueError('Unknown attack name {}.'.format(name))

    attack.goal_function.query_budget = max_queries

    return attack


def cmd_args_to_yaml(cmd_args, outfile_name, ignore_list=[]):
    """Takes cmd_args, an argparse.Namespace object, and writes the values to a file
    in YAML format. Some parameter values might not need to be saved, so you can
    pass a list of parameter names as the ignore_list, and the values for these
    parameter names will not be saved to the YAML file.
    """
    cmd_args_dict = vars(cmd_args)
    with open(outfile_name, 'w') as yaml_outfile:
        for parameter, value in cmd_args_dict.items():
            # don't write the parameter value if parameter in the
            # ignore list or the value of the parameter is None
            if parameter in ignore_list or value is None:
                continue
            else:
                # write boolean values as 1's and 0's
                if isinstance(value, bool):
                    value = int(value)
                yaml_outfile.write(f'{parameter}: {value}\n')


class CSVLoggerAttack(CSVLogger):
    def __init__(self, cmd_args):
        if cmd_args.exp_name is None:
            if cmd_args.log_output_type == 'darpa':
                unique_id = time.strftime("%Y%m%d-%H%M%S")  # datetime.now().strftime("%d-%m-%Y")  # _%H-%M")  # '1'
                # experiment_basename = \
                #     f'{cmd_args.model_name}-{cmd_args.dataset_name}-{cmd_args.attack_name}_{unique_id}'
            else:
                unique_id = '2'
            experiment_basename = f'{cmd_args.model_name}_{unique_id}'
        else:
            experiment_basename = cmd_args.exp_name
        # safely make the experiment output directory if it doesn't exist
        dataset_basename = os.path.basename(os.path.splitext(cmd_args.dataset_name)[0])
        output_dir = os.path.join(cmd_args.dir_out,
                                  'attacks',
                                  cmd_args.task_name,
                                  dataset_basename,
                                  cmd_args.model_name,
                                  cmd_args.attack_name,
                                  experiment_basename)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        filename = os.path.join(output_dir, 'results.csv')
        super(CSVLoggerAttack, self).__init__(filename=filename, color_method="file")
        self.resume = cmd_args.resume
        self.attack_name = cmd_args.attack_name
        self.attack_toolchain = cmd_args.attack_toolchain
        self.task_name = cmd_args.task_name
        self.config_filename = os.path.join(output_dir, 'config.yml')
        self.cmd_args = cmd_args
        self.output_dir = output_dir

        # set output format
        if cmd_args.log_output_type == 'full':
            self.log_attack_result = self._log_attack_result_all_data
        else:
            self.log_attack_result = self._log_attack_result_darpa_data

        # try to load output file and get position in dataset where we left off
        if self.resume and cmd_args.log_output_type == 'full':
            try:
                self.df = pd.read_csv(filename)
                self.start_index = int(self.df.iloc[-1].dataset_index + 1)
            except FileNotFoundError:
                sys.stderr.write(f'File with name {filename} not found. Creating new file\n')
                self.start_index = 0
        else:
            self.start_index = 0
            # if we aren't resuming, then the filename should be made to be different than
            # all other files in the directory where we are saving the file
            num = 1
            while os.path.isfile(self.filename):
                basename, ext = os.path.splitext(self.filename)[-2:]
                file_num_idx = basename.find('__')
                if file_num_idx != -1:
                    basename = basename[:file_num_idx] + f'__{num}'
                else:
                    basename += f'__{num}'
                self.filename = basename + ext
                num += 1

        # write command args to YAML file
        ignore_list = ['dir_model', 'dir_dataset', 'dir_out', 'resume', 'log_freq', 'device']
        ignore_list = []
        cmd_args_to_yaml(cmd_args, outfile_name=self.config_filename, ignore_list=ignore_list)

    def _log_attack_result_all_data(self, result, dataset_index, total_attack_time, color=True):
        # original_text, perturbed_text = result.diff_color(self.color_method)
        original_text, perturbed_text = result.diff_color(color_method='file' if color else None)
        original_text = original_text.replace("\n", AttackedText.SPLIT_TOKEN)
        perturbed_text = perturbed_text.replace("\n", AttackedText.SPLIT_TOKEN)
        result_type = result.__class__.__name__.replace("AttackResult", "")
        row = {
            "dataset_index": int(dataset_index),
            "original_text": original_text,
            "perturbed_text": perturbed_text,
            "original_score": result.original_result.score,
            "perturbed_score": result.perturbed_result.score,
            "original_output": result.original_result.output,
            "perturbed_output": result.perturbed_result.output,
            "ground_truth_output": result.original_result.ground_truth_output,
            "num_queries": int(result.num_queries),
            "total_attack_time": total_attack_time,
            "result_type": result_type,
        }
        if self.attack_name == 'conceal':
            try:
                row['target_word'] = result.original_result.target_word
            except AttributeError:
                row['target_word'] = "NA"
        self.df = self.df.append(row, ignore_index=True)
        self._flushed = False

    def _log_attack_result_darpa_data(self, result, _, __, color=True):
        original_text, perturbed_text = result.diff_color(color_method='file' if color else None)
        original_text = original_text.replace("\n", AttackedText.SPLIT_TOKEN)
        perturbed_text = perturbed_text.replace("\n", AttackedText.SPLIT_TOKEN)
        row = {
            "perturbed_text": perturbed_text,
            "perturbed_output": result.perturbed_result.output,
            "task_name": self.task_name,
            "attack_name": self.attack_name,
            "attack_toolchain": self.attack_toolchain,
            "original_text": original_text,
            "config_file": self.config_filename
        }
        self.df = self.df.append(row, ignore_index=True)
        self._flushed = False

    def get_start_index(self):
        return self.start_index
