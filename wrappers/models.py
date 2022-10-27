from transformers import PegasusTokenizer, PegasusForConditionalGeneration, PegasusConfig
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig


class PEGASUS:
    def __init__(self, dataset='cnn_dailymail', cache_dir='', device=None):
        model_name = 'google/pegasus-' + dataset
        model = PegasusForConditionalGeneration(PegasusConfig())
        self.model = model.from_pretrained(model_name, cache_dir=cache_dir).to(device).eval()
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.device = device
        self.dataset = dataset

    def __call__(self, text_list):
        input_ids_list = self.tokenizer(text_list, truncation=True, return_tensors='pt',
                                        padding=True)['input_ids'].to(self.device)
        output_ids_list = self.model.generate(input_ids_list,
                                              min_length=0 if self.dataset == 'gigaword' else None)
        outputs_list = self.tokenizer.batch_decode(output_ids_list, skip_special_tokens=True)

        return outputs_list


class BART:
    def __init__(self, dataset='cnn_dailymail', cache_dir='', device=None):
        if dataset == 'gigaword':
            model_name = 'a1noack/bart-large-gigaword'
        else:
            model_name = 'facebook/bart-large-' + ('cnn' if dataset == 'cnn_dailymail' else dataset)
        model = BartForConditionalGeneration(BartConfig())
        self.model = model.from_pretrained(model_name, cache_dir=cache_dir).to(device).eval()
        self.tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir=cache_dir, bos_token='<s>',
                                                       eos_token='</s>', sep_token='</s>')
        self.device = device
        self.dataset = dataset
        self.max_input_length = 128 if self.dataset == 'gigaword' else None

    def __call__(self, text_list):
        input_ids_list = self.tokenizer(text_list, truncation=True, max_length=self.max_input_length,
                                        return_tensors='pt', padding=True)['input_ids'].to(self.device)
        # input_ids_list = self.tokenizer(text_list, padding=True, return_tensors='pt')['input_ids'].to(self.device)
        output_ids_list = self.model.generate(input_ids_list, min_length=0 if self.dataset == 'gigaword' else None)
        outputs_list = self.tokenizer.batch_decode(output_ids_list, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)

        return outputs_list
