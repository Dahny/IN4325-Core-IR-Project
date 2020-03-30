# FINAL_HEADERS = ['query_id','query','table_id','row','col','nul',
#   'in_link','out_link','pgcount','tImp','tPF','leftColhits','SecColhits',
#   'bodyhits','PMI','qInPgTitle','qInTableTitle','yRank','csr_score','idf1',
#   'idf2','idf3','idf4','idf5','idf6','max','sum','avg','sim','emax','esum','eavg',
#   'esim','cmax','csum','cavg','csim','remax','resum','reavg','resim','query_l','rel']

from bs4 import BeautifulSoup
import requests
import re
import wikipedia
import json
import math
import wikipediaapi
from nltk import word_tokenize
from preprocessing.utils import preprocess_string
import pageviewapi

wiki = wikipediaapi.Wikipedia('en', extract_format=wikipediaapi.ExtractFormat.HTML)

# Load lookup dictionaries
with open('dictionaries/words_page_titles.json') as f:
    dict_page_titles = json.load(f)
    f.close()
with open('dictionaries/words_section_titles.json') as f:
    dict_section_titles = json.load(f)
    f.close()
with open('dictionaries/words_captions.json') as f:
    dict_captions = json.load(f)
    f.close()
with open('dictionaries/words_headers.json') as f:
    dict_headers = json.load(f)
    f.close()
with open('dictionaries/words_data.json') as f:
    dict_data = json.load(f)
    f.close()
with open('dictionaries/entities_to_information.json') as f:
    dict_information = json.load(f)
    f.close()

def compute_baseline_features(data_table, query_col='query', table_col='raw_table_data'):
    """
    Compute all features regarded as baseline features in the paper
    :param data_table:
    :param query_col:
    :param table_col:
    :return:
    """

    n_documents = 100

    # Query features
    query_features = {
        'query_l': query_length,
        'idf1': idf_page_title,
        'idf2': idf_section_title,
        'idf3': idf_table_caption,
        'idf4': idf_table_heading,
        'idf5': idf_table_body,
        'idf6': idf_catch_all,
     }
    for k, v in query_features.items():
        data_table[k] = data_table[query_col].map(lambda x: v(x, n_documents))
    
    # Table featuresCreation
    table_features = {
        'row': number_of_rows,
        'col': number_of_columns,
        'nul': number_of_null,
        'in_link': number_of_in_links,
        'out_link': number_of_out_links,
        'tImp': table_importance,
        'tPF': page_fraction,
        'PMI': pmi,
        'pgview': average_page_view,
    }
    for k, v in table_features.items():
        data_table[k] = data_table[table_col].map(lambda x: v(x, n_documents))

    # Query-table featuresCreation
    query_table_fatures = {
        'leftColhits': term_frequency_query_in_left_column,
        'SecColhits': term_frequency_query_in_second_column,
        'bodyhits': term_frequency_query_in_table_body,
        'qInPgTitle': ratio_query_terms_in_page_title,
        'qInTableTitle': ratio_query_terms_in_table_title,
        'yRank': y_rank,
        'csr_score': mlm_similarity,
    }
    for k, v in query_table_fatures.items():
        data_table[k] = data_table.apply(lambda x: v(x[query_col], x[table_col]), axis=1)

    return data_table


# Query featuresCreation

def query_length(query, _):
    """
    Takes the query and returns the length
    :param query:
    :param _:
    :return:
    """
    return len(query.split(' '))


def idf_page_title(query, n):
    """
    Takes the query and returns the sum of the IDF scores of the words in the page titles
    :param query:
    :param n:
    :return:
    """
    preprocessed_query = preprocess_string(query)
    final_idf = 0
    for term in preprocessed_query:
        if term in dict_page_titles:
            final_idf += compute_idf_t(n, len(dict_page_titles[term]))
    return final_idf


def idf_section_title(query, n):
    """
    Takes the query and returns the sum of the IDF scores of the words in the section titles
    :param query:
    :param n:
    :return:
    """
    preprocessed_query = preprocess_string(query)
    final_idf = 0
    for term in preprocessed_query:
        if term in dict_section_titles:
            final_idf += compute_idf_t(n, len(dict_section_titles[term]))
    return final_idf


def idf_table_caption(query, n):
    """
    Takes the query and returns the sum of the IDF scores of the words in the table captions
    :param query:
    :param n:
    :return:
    """
    preprocessed_query = preprocess_string(query)
    final_idf = 0
    for term in preprocessed_query:
        if term in dict_captions:
            final_idf += compute_idf_t(n, len(dict_captions[term]))
    return final_idf


def idf_table_heading(query, n):
    """
    Takes the query and returns the sum of the IDF scores of the words in the table headings
    :param query:
    :param n:
    :return:
    """
    preprocessed_query = preprocess_string(query)
    final_idf = 0
    for term in preprocessed_query:
        if term in dict_headers:
            final_idf += compute_idf_t(n, len(dict_headers[term]))
    return final_idf


def idf_table_body(query, n):
    """
    Takes the query and returns the sum of the IDF scores of the words in the table bodies
    :param query:
    :param n:
    :return:
    """
    preprocessed_query = preprocess_string(query)
    final_idf = 0
    for term in preprocessed_query:
        if term in dict_data:
            final_idf += compute_idf_t(n, len(dict_data[term]))
    return final_idf


def idf_catch_all(query, n):
    """
    Takes the query and returns the sum of the IDF scores of the words in the all text of the tables
    :param query:
    :param n:
    :return:
    """
    preprocessed_query = preprocess_string(query)
    final_idf = 0
    for term in preprocessed_query:
        if term in dict_page_titles:
            final_idf += compute_idf_t(n, len(dict_page_titles[term]))
        if term in dict_section_titles:
            final_idf += compute_idf_t(n, len(dict_section_titles[term]))
        if term in dict_captions:
            final_idf += compute_idf_t(n, len(dict_captions[term]))
        if term in dict_headers:
            final_idf += compute_idf_t(n, len(dict_headers[term]))
        if term in dict_data:
            final_idf += compute_idf_t(n, len(dict_data[term]))
    return final_idf


def compute_idf_t(n, dft):
    return math.log(n / dft)


# Table featuresCreation

def number_of_rows(table, _):
    """
    Takes the table and returns the number of rows
    :param table:
    :return:
    """
    return table['numDataRows']


def number_of_columns(table, _):
    """
    Takes the table and return the number of columns
    :param table:
    :return:
    """
    return table['numCols']


def number_of_null(table, _):
    """
    Takes the table and returns the number of empty cells
    :param table:
    :return:
    """
    nulls = 0
    for row in table['data']:
        for cell in row:
            if cell == "":
                nulls += 1
    return nulls


def number_of_in_links(table, _):
    """
    Takes the table and returns the number of inlinks to the page embedding the table
    :param table:
    :return:
    """
    page = wiki.page(table['pgTitle'])
    return len(page.backlinks)


def number_of_out_links(table, _):
    """
    Takes the table and returns the number of outlinks to the page embedding the table
    :param table:
    :return:
    """
    page = wiki.page(table['pgTitle'])
    return len(page.links)


def table_importance(table, _):
    """
    Takes the table and returns the inverse of the number of tables on the page
    :param table:
    :return:
    """
    page = wiki.page(table['pgTitle'])
    website_url = requests.get(page.fullurl).text
    soup = BeautifulSoup(website_url, 'html.parser')
    n_wiki_tables = len(soup.find_all('table', {'class': 'wikitable'}))

    return 1.0 / n_wiki_tables


def page_fraction(table, _):
    """
    Takes the table and returns the inverse of the number of tables on the page
    :param table:
    :return:
    """
    page = wiki.page(table['pgTitle'])
    website_url = requests.get(page.fullurl).text
    soup = BeautifulSoup(website_url, 'html.parser')
    paragraphs = soup.find_all('p')
    n_words = 0
    for paragraph in paragraphs:
        # Remove <..> and [..] from the paragraph
        simple_text = re.sub("[\<\[].*?[\>\]]", "", str(paragraph))
        # Count the number of words in the paragraph
        n_words += len(simple_text.split())

    table_size = table['numCols'] * table['numDataRows']
    return table_size / n_words


def pmi(table, n):
    """
    Takes the table and returns the ACSDb-based schema coherency score
    :param table:
    :param n:
    :return:
    """
    with open(main_path_name + '/dictionaries/dict_headers.json') as file:
        dict_headers = json.load(file)
        file.close()

    average_pmi = 0
    counter = 0
    preprocessed_headers = list(map(lambda x: ' '.join(preprocess_string(x)), table['title']))
    for i in range(len(preprocessed_headers) - 1):
        for j in range(i + 1, len(preprocessed_headers)):
            counter += 1
            average_pmi += compute_pmi(preprocessed_headers[i], preprocessed_headers[j], n, dict_headers)
    return average_pmi / counter


def compute_pmi(term_a, term_b, n, dictionary):
    """
    Compute the Pointwise Mutual Information between term a, and term b
    :param term_a:
    :param term_b:
    :param n:
    :param dictionary:
    :return:
    """
    p_a = 0
    p_b = 0
    tables_a = []
    tables_b = []
    if term_a in dictionary:
        tables_a = dictionary[term_a]
        p_a = len(tables_a) / n
    if term_b in dictionary:
        tables_b = dictionary[term_b]
        p_b = len(tables_b) / n
    p_a_b = len(set(tables_a).intersection(tables_b)) / n
    return math.log(p_a_b / (p_a * p_b))


def average_page_view(table):
    """
    Takes the table and returns the average page views of the Wikipedia page over a given time interval
    :param table:
    :return:
    """
    start_date = '20200101'     # January 1 2020
    end_date = '20200326'       # March 26 2020
    n_of_page_views = 0
    page_views = pageviewapi.per_article('en.wikipedia', table['pgTitle'], start_date, end_date,
                                         access='all-access', agent='all-agents', granularity='daily')
    for article in page_views['items']:
        n_of_page_views += article['views']
    return n_of_page_views / 86


# Query-table featuresCreation

def term_frequency_query_in_left_column(query, table):
    """
    Total query term frequency in the leftmost column cells
    :param query:
    :param table:
    :return:
    """
    tokenized_query = word_tokenize(query.lower())
    tokenized_first_column = list(map(lambda x: preprocess_string(x), table['data'][:][0]))
    number_found = 0
    for query_token in tokenized_query:
        if query_token in tokenized_first_column:
            number_found += 1
    return number_found


def term_frequency_query_in_second_column(query, table):
    """
    Total query term frequency in second-to-leftmost column cells
    :param query:
    :param table:
    :return:
    """
    tokenized_query = word_tokenize(query.lower())
    tokenized_second_column = list(map(lambda x: preprocess_string(x), table['data'][:][1]))
    number_found = 0
    for query_token in tokenized_query:
        if query_token in tokenized_second_column:
            number_found += 1
    return number_found


def term_frequency_query_in_table_body(query, table):
    """
    Total query term frequency in the table body
    :param query:
    :param table:
    :return:
    """
    tokenized_query = word_tokenize(query.lower())
    number_found = 0
    for row in table['data']:
        data_row = list(map(lambda x: preprocess_string(x), row))
        for cell in data_row:
            print("cell ", cell)
            for query_token in tokenized_query:
                if query_token in cell:
                    number_found += 1
    return number_found


def ratio_query_terms_in_page_title(query, table):
    """
    Ratio of the number of query tokens found in page title to total number of tokens
    :param query:
    :param table:
    :return:
    """
    tokenized_query = preprocess_string(query)
    tokenized_page_title = preprocess_string(table['pgTitle'])
    number_found = 0
    for query_token in tokenized_query:
        if query_token in tokenized_page_title:
            number_found += 1

    return number_found / len(tokenized_query)


def ratio_query_terms_in_table_title(query, table):
    """
    Ratio of the number of query tokens found in table title to total number of tokens
    :param query:
    :param table:
    :return:
    """
    tokenized_query = preprocess_string(query)
    tokenized_table_title = preprocess_string(table['caption'])
    number_found = 0
    for query_token in tokenized_query:
        if query_token in tokenized_table_title:
            number_found += 1

    return number_found / len(tokenized_query)


def y_rank(query, table):
    """
    Rank of the table’s Wikipedia page in Web search engine results for the query
    :param query:
    :param table:
    :return:
    """
    max_result = 50
    wikipedia.set_lang('en')
    wikiterm = wikipedia.search(query)
    for idx, term in enumerate(wikiterm[0:max_result]):
        wikipage = wikipedia.page(term)
        if wikipage.title == table['pgTitle']:
            return idx
    return max_result + 1


def mlm_similarity(query, table):
    ''' Language modeling score between query and multi-field document repr. of the table '''
    return 0
