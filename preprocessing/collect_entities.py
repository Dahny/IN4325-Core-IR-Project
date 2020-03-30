import json
import requests
import re

import wikipediaapi
from tqdm import tqdm
import pageviewapi
from bs4 import BeautifulSoup

from utils import preprocess_string, extract_entities_from_wikipedia_string, find_core_column, list_to_ngrams, write_dictionary_to_file


wiki = wikipediaapi.Wikipedia('en')


def collect_entity_information(table_file, query_file, output_folder, cont=False):
    table_entities, table_to_entities = find_all_entities_in_tables(table_file)
    query_entities, query_to_entities =  find_all_entities_in_queries(query_file)
    entities = list(set(table_entities + query_entities))

    write_dictionary_to_file(table_to_entities, output_folder + '/table_to_entities.json')
    write_dictionary_to_file(query_to_entities, output_folder + '/query_to_entities.json')

    entities_to_information = {}

    for entity in tqdm(entities):
        page = wiki.page(entity)
        if page.exists():
            new_entity = {}
            new_entity['inlinks'] = list(page.backlinks.keys())
            new_entity['outlinks'] = list(page.links.keys())
            new_entity['categories'] = list(page.categories.keys())
            new_entity['page_views'] = average_page_view(entity)
            new_entity['nr_of_tables'], new_entity['nr_of_words'] = nr_of_tables_and_words(page)
            entities_to_information[entity] = new_entity
    
    write_dictionary_to_file(entities_to_information, output_folder + '/entities_to_information.json')



def find_all_entities_in_tables(table_file):
    entities = []
    table_to_entities = {}
    with open(table_file) as json_file:
        wiki_tables = json.load(json_file)
        for table_id, table in wiki_tables.items():
            table_to_entities[table_id] = []
            table_to_entities[table_id] += extract_entities_from_wikipedia_string(table['pgTitle'])
            table_to_entities[table_id] += extract_entities_from_wikipedia_string(table['caption'])
            if len(table['data']) > 0:
                core_column = find_core_column(table['data'])
                concat_core_column_string = ''
                for row in table['data']:
                    concat_core_column_string += ' ' + row[core_column]
                table_to_entities[table_id] += extract_entities_from_wikipedia_string(concat_core_column_string)
            table_to_entities[table_id] = list(set(table_to_entities[table_id]))
            entities += table_to_entities[table_id]
    return entities, table_to_entities


def find_all_entities_in_queries(query_file):
    query_to_ngrams = {}

    with open(query_file) as f:
        for line in f:
            splitted = line.split(' ')
            query_to_ngrams[splitted[0]] = list_to_ngrams(splitted[1:])

    query_to_entities = {}
    entities = []

    for k, v in query_to_ngrams.items():
        query_entities = [check_if_entity(i) for i in v if check_if_entity(i) is not None]
        query_to_entities[k] = list(set(query_entities))
        entities += query_entities

    return entities, query_to_entities
    

def check_if_entity(s):
    page = wiki.page(s)
    if page.exists():
        res = page.fullurl.split('/')[-1]
        if res.lower().replace('_', ' ') == s:
            return res
        else:
            return
    else:
        return


def average_page_view(entity):
    """
    Takes the table and returns the average page views of the Wikipedia page over a given time interval
    :param table:
    :return:
    """
    start_date = '20200101'     # January 1 2020
    end_date = '20200326'       # March 26 2020
    n_of_page_views = 0
    try:
        page_views = pageviewapi.per_article('en.wikipedia', entity, start_date, end_date,
                                         access='all-access', agent='all-agents', granularity='daily')
        for article in page_views['items']:
            n_of_page_views += article['views']
    except:
        n_of_page_views = 0

    return n_of_page_views / 86


def nr_of_tables_and_words(page):
    website_url = requests.get(page.fullurl).text
    soup = BeautifulSoup(website_url, 'html.parser')
    n_wiki_tables = len(soup.find_all('table', {'class': 'wikitable'}))

    paragraphs = soup.find_all('p')
    n_words = 0
    for paragraph in paragraphs:
        # Remove <..> and [..] from the paragraph
        simple_text = re.sub("[\<\[].*?[\>\]]", "", str(paragraph))
        # Count the number of words in the paragraph
        n_words += len(simple_text.split())

    return n_wiki_tables, n_words


if __name__ == '__main__':
    table_file = '../data/raw_table_data.json'
    query_file = '../data/queries.txt'
    output_folder = '../dictionaries'

    collect_entity_information(table_file, query_file, output_folder)