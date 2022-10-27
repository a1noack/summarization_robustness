import random
import re
from rouge_score import rouge_scorer
import string

from textattack.shared import Attack, AttackedText
from textattack.attack_recipes import AttackRecipe
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification
)
from textattack.search_methods import (
    GreedyWordSwapWIR,
    GreedySearch
)
from textattack.transformations import (
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapNeighboringCharacterSwap,
    WordSwapWordNet,
    Transformation,
    WordSwapInflections,
    WordSwapContract,
    WordSwapExtend,
    WordSwapChangeNumber
)
from textattack.goal_functions.text.text_to_text_goal_function import TextToTextGoalFunction

from seq2seq.summarization.utils.string_utils import clean_str


# ========== GOAL FUNCTIONS ==========

class NoGoal(TextToTextGoalFunction):
    """A seq2seq model goal function that is satisfied as soon as
    a transformation has occurred once."""
    def __init__(self, model_wrapper, n_perturbations, *args, **kwargs):
        super().__init__(model_wrapper, *args, **kwargs)
        self.n_perturbations = n_perturbations

    def init_attack_example(self, attacked_text, ground_truth_output):
        """Called before attacking ``attacked_text`` to 'reset' the goal
        function and set properties for this example."""
        self.transformation_number = 0  # the number of transformations attempted when perturbing this sample
        self.initial_attacked_text = attacked_text
        self.ground_truth_output = ground_truth_output
        self.num_queries = 0
        result, _ = self.get_result(attacked_text, check_skip=True)
        return result, _

    def _is_goal_complete(self, model_output, attacked_text):
        is_complete = False if self.transformation_number == 0 else True
        self.transformation_number += 1

        ##### NEW #######
        is_complete = True if attacked_text.n_steps_from_orig >= self.n_perturbations else False
        print(attacked_text, attacked_text.n_steps_from_orig, self.n_perturbations)
        ##### END NEW ######

        return is_complete

    def _get_score(self, model_output, attacked_text):
        # so apparently, _get_score here is called twice initially,
        # whereas _is_goal_complete is only called once on the original sample;
        # hence the two separate threshold values, 2 here and 0 in _is_goal_complete
        score = 0 if self.transformation_number <= 2 else random.random()

        ##### NEW #####
        score = 0 if attacked_text.n_steps_from_orig == 0 else random.random() * attacked_text.n_steps_from_orig
        ##### END NEW #####

        return score


class MinimizeRouge(TextToTextGoalFunction):
    """
    :param rouge_metric: 1, 2, or 'Lsum'
    """

    EPS = 1e-10

    def __init__(self, *args, target_rouge=0.0, rouge_metric=2, **kwargs):
        self.target_rouge = target_rouge
        assert rouge_metric in [1, 2, 'Lsum'], 'Invalid ROUGE metric'
        self.rouge_metric = f'rouge{rouge_metric}'
        self.rouge_scorer = rouge_scorer.RougeScorer([self.rouge_metric], use_stemmer=True)
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, _):
        rouge_score = 1 - self._get_score(model_output, _)
        return rouge_score <= (self.target_rouge + MinimizeRouge.EPS)

    def _get_score(self, model_output, _):
        # model_output_at = AttackedText(model_output)
        # ground_truth_at = AttackedText(self.ground_truth_output)
        ground_truth = clean_str(self.ground_truth_output)
        model_output = clean_str(model_output)
        rouge_score = self.rouge_scorer.score(ground_truth, model_output)[self.rouge_metric].fmeasure
        return 1 - rouge_score

    def extra_repr_keys(self):
        if self.maximizable:
            return ["maximizable"]
        else:
            return ["maximizable", "target_rouge"]


# ========== TRANSFORMATIONS ==========

class PunctuationSwap(Transformation):
    """Randomly chooses a white space position in the current text to
    swap out for a random sentence-ending punctuation mark, or vice-versa"""
    def __init__(self, split_sent, num_candidates=1, cased=True):
        self.num_candidates = num_candidates
        self.sent_end_punc = ['.', '?', '!']
        self.sent_concat_chars = [',', ' ', '—', ';']
        self.cased = cased
        self.split_sent = split_sent  # if True, split sentences, if False, concatenate sentences

    def _get_transformations(self, current_text, _):
        if self.split_sent:
            return self._get_transformations_split(current_text, _)
        else:
            return self._get_transformations_concat(current_text, _)

    def _get_transformations_split(self, current_text, _):
        transformed_texts = []
        current_text = current_text.text
        space_idxs = [match.start() for match in re.finditer(r' ', current_text)]  # all indices in text that are spaces

        # filter out invalid indices
        candidate_idxs = []
        for idx in space_idxs:
            if current_text[idx - 1] not in self.sent_end_punc and \
                    not current_text[idx + 1].isupper() and \
                    current_text[idx + 1].isalpha():
                candidate_idxs.append(idx)
        try:
            # randomly choose num_candidates indices
            candidate_idxs = random.sample(candidate_idxs, self.num_candidates)
        except ValueError:
            return transformed_texts

        # create AttackedText candidates
        for idx in candidate_idxs:
            punctuation = random.choice(self.sent_end_punc)
            try:
                next_letter = current_text[idx + 1].upper() if self.cased else current_text[idx + 1]
                candidate_text = current_text[:idx] + punctuation + ' ' + next_letter + current_text[idx + 2:]
                transformed_texts.append(AttackedText(candidate_text))
            except IndexError:
                pass

        return transformed_texts

    def _get_transformations_concat(self, current_text, _):
        transformed_texts = []
        current_text = current_text.text
        escaped_punc_regex = re.compile('|'.join([("\\" + c) for c in self.sent_end_punc]))
        punc_idxs = [match.start() for match in re.finditer(escaped_punc_regex, current_text)]

        # filter out invalid indices
        candidate_idxs = []
        for idx in punc_idxs:
            try:
                if current_text[idx - 1].islower() and current_text[idx + 1] == ' ' and current_text[idx + 2].isalpha():
                    if (self.cased and current_text[idx + 2].isupper()) or not self.cased:
                        candidate_idxs.append(idx)
            except IndexError:
                pass
        try:
            # randomly choose num_candidates indices
            candidate_idxs = random.sample(candidate_idxs, self.num_candidates)
        except ValueError:
            return transformed_texts

        # create AttackedText candidates
        for idx in candidate_idxs:
            join_char = random.choice(self.sent_concat_chars)
            try:
                next_letter = current_text[idx + 2].lower() if self.cased else current_text[idx + 2]
                candidate_text = current_text[:idx] + join_char + ' ' + next_letter + current_text[idx + 3:]
                transformed_texts.append(AttackedText(candidate_text))
            except IndexError:
                pass

        return transformed_texts

    @property
    def deterministic(self):
        return False


class InsertNonsence(Transformation):
    """Insert random string of characters into text – at the beginning,
    the end, or somewhere in the middle.
    :param where: 'start', to prepend nonsense to start of text, 'end', to append to end
        of text, or 'middle' to insert randomly into middle of text
    """
    def __init__(self, where='start', nonsense_length=5, cased=True, num_candidates=1):
        assert where in ['start', 'end', 'middle'], 'Invalid nonsense insertion location!'
        self.where = where
        self.nonsense_length = nonsense_length
        self.num_candidates = num_candidates
        self.chars = string.digits + string.ascii_lowercase + string.punctuation + ' '
        self.chars = self.chars + (string.ascii_uppercase if cased else '')

    def _get_transformations(self, current_text, _):
        current_text = current_text.text
        transformed_texts = []
        for i in range(self.num_candidates):
            nonsense = ''.join([random.choice(self.chars) for _ in range(self.nonsense_length)])
            candidate_text = None
            if self.where == 'start':
                candidate_text = nonsense + ' ' + current_text
            elif self.where == 'end':
                candidate_text = current_text + ' ' + nonsense
            elif self.where == 'middle':
                try:
                    insert_idx = random.choice([match.start() for match in re.finditer(r' ', current_text)])
                    candidate_text = current_text[:insert_idx] + ' ' + nonsense + current_text[insert_idx:]
                except IndexError:
                    pass

            if candidate_text:
                transformed_texts.append(AttackedText(candidate_text))

        return transformed_texts


class RandomSentenceSwap(Transformation):
    """Randomly swaps position of two sentences in the input"""
    def __init__(self, num_candidates=1):
        self.num_candidates = num_candidates

    @staticmethod
    def _find_sentences(text):
        sentence_start_idxs = [match.end()-1 for match in re.finditer(r'(\.|\!|\?) [A-Z]', text)]
        sentence_ranges = []
        prev_idx = None
        for cur_idx in sentence_start_idxs:
            if prev_idx is not None:
                sentence_ranges.append((prev_idx, cur_idx))
            prev_idx = cur_idx
        return sentence_ranges

    def _get_transformations(self, current_text, _):
        transformed_texts = []
        current_text = current_text.text

        sentence_ranges = self._find_sentences(current_text)
        if len(sentence_ranges) < 2:
            return transformed_texts

        for i in range(self.num_candidates):
            range1, range2 = random.sample(sentence_ranges, 2)
            if range1[0] > range2[0]:
                temp = range1
                range1 = range2
                range2 = temp
            sentence1 = current_text[range1[0]: range1[1]]
            sentence2 = current_text[range2[0]: range2[1]]
            candidate_text = current_text[:range1[0]] + sentence2 + current_text[range1[1]:range2[0]] + \
                             sentence1 + current_text[range2[1]:]
            transformed_texts.append(AttackedText(candidate_text))

        return transformed_texts


# ========== ATTACK RECIPES ==========

class AdjacentCharSwap(AttackRecipe):
    """Randomly chooses a word and swaps two internal and adjacent characters"""
    @staticmethod
    def build(model_wrapper, n_perturbations=1):
        goal_function = NoGoal(model_wrapper, n_perturbations=n_perturbations)
        transformation = WordSwapNeighboringCharacterSwap(random_one=False, skip_first_char=True, skip_last_char=True)
        constraints = [RepeatModification(), StopwordModification(),
                       MaxWordsPerturbed(max_num_words=n_perturbations)]
        search_method = GreedyWordSwapWIR(wir_method="random")

        return Attack(goal_function, constraints, transformation, search_method)


class WordSynonymSwap(AttackRecipe):
    """Randomly chooses a word and swaps two internal and adjacent characters"""
    @staticmethod
    def build(model_wrapper, synonym_method='wordnet', n_perturbations=1):
        goal_function = NoGoal(model_wrapper, n_perturbations=n_perturbations)
        if synonym_method == 'wordnet':
            transformation = WordSwapWordNet()
        elif synonym_method == 'embedding':
            transformation = WordSwapEmbedding()
        else:
            raise ValueError(f'{synonym_method} is not a valid transformation type!')
        constraints = [RepeatModification(), StopwordModification(),
                       MaxWordsPerturbed(max_num_words=n_perturbations)]
        search_method = GreedyWordSwapWIR(wir_method="random")

        return Attack(goal_function, constraints, transformation, search_method)


class SentenceSplit(AttackRecipe):
    """Randomly chooses a space in text at which to split into two sentences"""
    @staticmethod
    def build(model_wrapper, n_perturbations=1):
        goal_function = NoGoal(model_wrapper, n_perturbations=n_perturbations)
        transformation = PunctuationSwap(split_sent=True, cased=model_wrapper.dataset != 'gigaword')
        constraints = []
        search_method = GreedyWordSwapWIR(wir_method="random")

        return Attack(goal_function, constraints, transformation, search_method)


class SentenceConcat(AttackRecipe):
    """Randomly join two sentences"""
    @staticmethod
    def build(model_wrapper, n_perturbations=1):
        goal_function = NoGoal(model_wrapper, n_perturbations=n_perturbations)
        transformation = PunctuationSwap(split_sent=False, cased=model_wrapper.dataset != 'gigaword')
        constraints = []
        search_method = GreedyWordSwapWIR(wir_method="random")

        return Attack(goal_function, constraints, transformation, search_method)


class RandomStringInsert(AttackRecipe):
    """Randomly inserts random string of characters into text"""
    @staticmethod
    def build(model_wrapper, n_perturbations=1):
        goal_function = NoGoal(model_wrapper, n_perturbations=n_perturbations)
        transformation = InsertNonsence(where='middle', cased=model_wrapper.dataset != 'gigaword')
        constraints = []
        search_method = GreedyWordSwapWIR(wir_method="random")

        return Attack(goal_function, constraints, transformation, search_method)


class SentenceSwap(AttackRecipe):
    """Randomly swaps the position of two sentences in the input"""
    @staticmethod
    def build(model_wrapper, n_perturbations=1):
        goal_function = NoGoal(model_wrapper, n_perturbations=n_perturbations)
        transformation = RandomSentenceSwap()
        constraints = []
        search_method = GreedyWordSwapWIR(wir_method="random")

        return Attack(goal_function, constraints, transformation, search_method)


class RougeAttack(AttackRecipe):
    """Optimizes for reduction in ROUGE.
    """
    @staticmethod
    def build(model_wrapper, target_rouge=0):
        goal_function = MinimizeRouge(model_wrapper, target_rouge=target_rouge)
        transformation = CompositeTransformation([WordSwapInflections(),
                                                  WordSwapContract(),
                                                  WordSwapWordNet(),
                                                  WordSwapExtend(),
                                                  # WordSwapChangeNumber()
                                                  ])
        constraints = [RepeatModification(), StopwordModification()]
        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)
