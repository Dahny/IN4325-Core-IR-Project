import json
from utils import preprocess_string, extract_entities_from_wikipedia_string, find_core_column, list_to_ngrams


def collect_entity_information(table_file, query_file, output_folder):
    entities = find_all_entities_in_tables(table_file)
    entities += find_all_entities_in_queries(query_file)


def find_all_entities_in_tables(table_file):
    entities = []
    with open(table_file) as json_file:
        wiki_tables = json.load(json_file)
        for table_id, table in wiki_tables.items():
            entities += extract_entities_from_wikipedia_string(table['pgTitle'])
            entities += extract_entities_from_wikipedia_string(table['caption'])
            if len(table['data']) > 0:
                core_column = find_core_column(table['data'])
                concat_core_column_string = ''
                for row in table['data']:
                    concat_core_column_string += ' ' + row[core_column]
                entities += extract_entities_from_wikipedia_string(concat_core_column_string)
    return entities


def find_all_entities_in_queries(query_file):
    queries = []

    with open(query_file) as f:
        for line in f:
            queries.append(line.split(' ')[1:])
    
    n_grams_of_queries = []

    n_grams_of_queries = [y for x in queries for y in list_to_ngrams(x)]

    print('Start n_grams')
    print(n_grams_of_queries)
    print('End n_grams')

    return []


if __name__ == '__main__':
    table_file = '../data/raw_table_data.json'
    query_file = '../data/queries.txt'
    output_folder = '../dictionaries'

    collect_entity_information(table_file, query_file, output_folder)