import re


def clean_str(s):
    # the rougeLsum scorer method expects new sentences to be demarked by new line characters
    s = re.sub("[.!?]", "\n", s)
    # PEGASUS CNN-Dailymail tokenizer outputs <n> string between sentences
    s = re.sub("<n>", "\n", s)
    # Gigaword tokenizer uses 'unk_3' token to mark end of output string
    s = re.sub("unk_3", "", s)
    return s