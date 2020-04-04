import re
import string
from nltk import word_tokenize
import json
import ast


def write_dictionary_to_file(dictionary, file_name):
    with open(file_name, 'w') as f:
        output = json.dumps(dictionary, indent=4)
        output = re.sub(r'",\s+', '", ', output)
        f.write(output) 
        f.close()


def to_csv_line(d, fields=['name', 'inlinks', 'outlinks', 'categories', 'page_views', 'nr_of_tables', 'nr_of_words']):
    line = ''
    for field in fields:
        line += str(d[field]) + '##;;##'
    return line[:-1] + '\n'


def from_csv_line(line, fields=['name', 'inlinks', 'outlinks', 'categories', 'page_views', 'nr_of_tables', 'nr_of_words']):
    splitted = line.split('##;;##')
    assert len(splitted) == len(fields)
    result = {}
    for i in range(len(splitted)):
        key = fields[i]
        value = splitted[i].strip().strip('##;;##')
        if key == 'inlinks' or key == 'outlinks' or key == 'categories':
            new_value = re.split("""', '|", "|', "|", '""", value)
            new_value = [re.sub("""\['|\["|"\]|'\]|\[|\]""", '', x.strip().replace(' ', '_')) for x in new_value]
            result[key] = new_value
        else:
            result[key] = value
    return result


def get_entity_to_information_dict(path, fields=['name', 'inlinks', 'outlinks', 'categories', 'page_views', 'nr_of_tables', 'nr_of_words']):
    result = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            d = from_csv_line(line, fields)
            result[d['name']] = d
    return result


def preprocess_string(s):
    pre_s = re.sub('[\(\[].*?[\|\)]', '', s.lower())
    pre_s = pre_s.translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(pre_s)


def extract_entities_from_wikipedia_string(s):
    entities = re.findall('(?<=\[).*?(?=\])', s)
    result = []
    for entity in entities:
        if '|' in entity:
            result.append(entity.split('|')[0])
    return result


def find_core_column(data):
    if len(data) is 0:
        return
    number_of_columns = len(data[0])
    entities_in_column = [0] * number_of_columns
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
