import re
import string
from nltk import word_tokenize


def write_dictionary_to_file(dictionary, file_name):
    with open(file_name, 'w') as f:
        json.dump(dictionary, f)
        f.close()


def preprocess_string(s):
    pre_s = re.sub('[\(\[].*?[\|\)]', '', s.lower())
    pre_s = pre_s.translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(pre_s)


def extract_entities_from_wikipedia_string(s):
    entities = re.findall('(?<=\[).*?(?=\])',s)
    result = []
    for entity in entities:
        result.append(entity.split('|')[0])
    return result


def find_core_column(data):
    if len(data) is 0:
        return
    number_of_colums = len(data[0])
    number_of_rows = len(data)
    entities_in_column = [0] * number_of_colums
    for row in data:
        for i, cell in enumerate(row):
            if len(extract_entities_from_wikipedia_string(cell)) > 0:
                entities_in_column[i] += 1
        return entities_in_column.index(max(entities_in_column))
    

def list_to_ngrams(words):
    words_incl_n_grams = []
    for N in range(1, len(words) + 1):
        words_incl_n_grams += [' '.join(words[i:i+N]).strip() for i in range(len(words)-N+1)]
    return words_incl_n_grams
