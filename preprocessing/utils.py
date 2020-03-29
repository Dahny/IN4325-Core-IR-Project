import re
import string
from nltk import word_tokenize


def preprocess_string(s):
    pre_s = re.sub('[\(\[].*?[\|\)]', '', s.lower())
    pre_s = pre_s.translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(pre_s)