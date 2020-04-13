import json
from preprocessing.utils import preprocess_string, write_dictionary_to_file


def create_dictionaries_from_wiki_tables(input_file, output_folder):
    dict_headers = {}
    dict_page_titles = {}
    dict_captions = {}
    dict_section_titles = {}
    dict_data = {}

    with open(input_file) as json_file:
        wiki_tables = json.load(json_file)
        for table_id, wiki_table in wiki_tables.items():
            preprocessed_page_title = preprocess_string(wiki_table['pgTitle'])
            list(map(lambda x: add_to_dict(dict_page_titles, x, table_id), preprocessed_page_title))

            preprocessed_section_title = preprocess_string(wiki_table['secondTitle'])
            list(map(lambda x: add_to_dict(dict_section_titles, x, table_id), preprocessed_section_title))

            preprocessed_caption = preprocess_string(wiki_table['caption'])
            list(map(lambda x: add_to_dict(dict_captions, str(x), table_id), preprocessed_caption))

            preprocessed_headers = [x for title in wiki_table['title'] for x in preprocess_string(title)]
            list(map(lambda x: add_to_dict(dict_headers, str(x), table_id), preprocessed_headers))

            preprocessed_data = list(map(lambda x: list(map(lambda y: preprocess_string(y), x)), wiki_table['data']))
            list(map(lambda x: list(map(lambda y: list(map(lambda z: add_to_dict(dict_data, z, table_id), y)), x)),
                     preprocessed_data))

    write_dictionary_to_file(dict_headers, output_folder + '/words_headers.json')
    write_dictionary_to_file(dict_page_titles, output_folder + '/words_page_titles.json')
    write_dictionary_to_file(dict_section_titles, output_folder + '/words_section_titles.json')
    write_dictionary_to_file(dict_captions, output_folder + '/words_captions.json')
    write_dictionary_to_file(dict_data, output_folder + '/words_data.json')


def add_to_dict(dictionary, word, table_id):
    if word not in dictionary:
        dictionary[word] = [table_id]
    else:
        if table_id not in dictionary[word]:
            dictionary[word].append(table_id)


if __name__ == '__main__':
    input_file = '../data/raw_table_data.json'
    output_folder = '../dictionaries'

    create_dictionaries_from_wiki_tables(input_file, output_folder)
