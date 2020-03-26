# FINAL_HEADERS = ['query_id','query','table_id','row','col','nul',
#   'in_link','out_link','pgcount','tImp','tPF','leftColhits','SecColhits',
#   'bodyhits','PMI','qInPgTitle','qInTableTitle','yRank','csr_score','idf1',
#   'idf2','idf3','idf4','idf5','idf6','max','sum','avg','sim','emax','esum','eavg',
#   'esim','cmax','csum','cavg','csim','remax','resum','reavg','resim','query_l','rel']

from bs4 import BeautifulSoup
import requests
import re
import wikipedia
import numpy as np
import wikipediaapi
from nltk import word_tokenize
wiki = wikipediaapi.Wikipedia('en', extract_format=wikipediaapi.ExtractFormat.HTML)


def compute_baseline_features(data_table, query_col='query', table_col='raw_table_data'):
    """
    Compute all features regarded as baseline features in the paper
    :param data_table:
    :param query_col:
    :param table_col:
    :return:
    """

    # Query features
    query_features_names = ['query_l', 'idf1', 'idf2', 'idf3', 'idf4', 'idf5', 'idf6']
    data_table[query_features_names[0]] = list(map(query_length, data_table[query_col]))
    idf_s = list(map(lambda x: idf(x, data_table[table_col]), data_table[query_col]))
    for i in range(1, len(query_features_names)):
        data_table[query_features_names[i]] = idf_s[:][i-1]
    
    # Table features
    table_features = {
        'row': number_of_rows,
        'col': number_of_columns,
        'nul': number_of_null,
        'in_link': number_of_in_links,
        'out_link': number_of_out_links,
        'tImp': table_importance,
        'tPF': page_fraction,
        'PMI': pmi,
        'pgcount': page_views,
    }
    for k, v in table_features.items():
        data_table[k] = data_table[table_col].map(v)

    # Query-table features
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


# Query features

def query_length(query, _):
    ''' Takes the query and returns the length '''
    return len(query.split(' '))


def idf(query, full_data_table):
    ''' Takes the query and returns the sum of the IDF scores of the words in the page titles'''
    ''' ( IDF_t = log ( N / df_t ) 
        Here N is the number of documents and df_t the number of documents containing word t 
        This is then summed for all terms in the query ''' 
    ''' Note: this is also dependent on data about all documents (/tables). 
    We still need to get this here somehow '''

    tokenized_query = word_tokenize(query.lower())
    counter_items_page_title = np.zeros(len(tokenized_query))
    counter_items_caption = np.zeros(len(tokenized_query))
    counter_items_header_titles = np.zeros(len(tokenized_query))
    counter_items_section_title = np.zeros(len(tokenized_query))
    counter_items_data = np.zeros(len(tokenized_query))

    for table in full_data_table:

        tokenized_page_title = word_tokenize(table['pgTitle'].lower())
        tokenized_caption = word_tokenize(table['caption'].lower())
        header_titles = list(map(lambda header: ' '.join(word_tokenize(header.lower())).replace('|', ' '),
                                 table['title']))
        tokenized_section_title = word_tokenize(table['secondTitle'].lower())

        for i in range(len(tokenized_query)):

            if tokenized_query[i] in tokenized_page_title:
                counter_items_page_title[i] += 1

            if tokenized_query[i] in tokenized_caption:
                counter_items_caption[i] += 1

            if tokenized_query[i] in header_titles:
                counter_items_header_titles[i] += 1

            if tokenized_query[i] in tokenized_section_title:
                counter_items_section_title[i] += 1

            for row in table['data']:
                data_row = list(map(lambda v: word_tokenize(v.lower().replace('|', ' ')), row))
                for cell in data_row:
                    print("cell ", cell)
                    if tokenized_query[i] in cell:
                        counter_items_data[i] += 1

    counter_items_all = counter_items_page_title + counter_items_caption + counter_items_header_titles + \
                        counter_items_section_title + counter_items_data

    IDF_t_page_title = compute_idf_t(len(full_data_table), counter_items_page_title)
    IDF_t_caption = compute_idf_t(len(full_data_table), counter_items_caption)
    IDF_t_header_title = compute_idf_t(len(full_data_table), counter_items_header_titles)
    IDF_t_section_title = compute_idf_t(len(full_data_table), counter_items_section_title)
    IDF_t_data = compute_idf_t(len(full_data_table), counter_items_data)
    IDF_t_all = compute_idf_t(len(full_data_table), counter_items_all)

    return [IDF_t_page_title, IDF_t_caption, IDF_t_header_title, IDF_t_section_title, IDF_t_data, IDF_t_all]


def compute_idf_t(n, dft_s):
    return np.sum(np.log(np.divide(n - dft_s + 0.5, dft_s + 0.5)))


# Table features

def number_of_rows(table):
    """
    Takes the table and returns the number of rows
    :param table:
    :return:
    """
    return table['numDataRows']


def number_of_columns(table):
    """
    Takes the table and return the number of columns
    :param table:
    :return:
    """
    return table['numCols']


def number_of_null(table):
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


def number_of_in_links(table):
    """
    Takes the table and returns the number of inlinks to the page embedding the table
    :param table:
    :return:
    """
    page = wiki.page(table['pgTitle'])
    return len(page.backlinks)


def number_of_out_links(table):
    """
    Takes the table and returns the number of outlinks to the page embedding the table
    :param table:
    :return:
    """
    page = wiki.page(table['pgTitle'])
    return len(page.links)


def table_importance(table):
    """
    Takes the table and returns the inverse of the number of tables on the page
    :param table:
    :return:
    """
    page = wiki.page(table['pgTitle'])
    website_url = requests.get(page.fullurl).text
    soup = BeautifulSoup(website_url, 'html.parser')
    n_wikitables = len(soup.find_all('table', {'class': 'wikitable'}))

    return 1.0 / n_wikitables


def page_fraction(table):
    """
    Takes the table and returns the inverse of the number of tables on the page
    :param table:
    :return:
    """
    page = wiki.page(table['pgTitle'])
    website_url = requests.get(page.fullurl).text
    soup = BeautifulSoup(website_url, 'html.parser')
    texts = soup.find_all('p')
    n_words = 0
    for text in texts:
        # Remove <..> and [..] from the paragraph
        simple_text = re.sub("[\<\[].*?[\>\]]", "", str(text))
        # Count the number of words in the paragraph
        n_words += len(simple_text.split())

    table_size = table['numCols'] * table['numDataRows']
    return table_size / n_words


def pmi(table):
    ''' Takes the table and returns the ACSDb-based schema coherency score '''
    return 0


def page_views(table):
    ''' Takes the table and returns the page views of the table '''
    return 0


# Query-table features

def term_frequency_query_in_left_column(query, table):
    """
    Total query term frequency in the leftmost column cells
    :param query:
    :param table:
    :return:
    """
    tokenized_query = word_tokenize(query.lower())
    tokenized_first_column = list(map(lambda v: word_tokenize(v.lower().replace('|', ' ')), table['data'][:][0]))
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
    tokenized_second_column = list(map(lambda v: word_tokenize(v.lower().replace('|', ' ')), table['data'][:][1]))
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
        data_row = list(map(lambda v: word_tokenize(v.lower().replace('|', ' ')), row))
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
    tokenized_query = word_tokenize(query.lower())
    tokenized_page_title = word_tokenize(table['pgTitle'].lower())
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
    tokenized_query = word_tokenize(query.lower())
    tokenized_table_title = word_tokenize(table['caption'].lower())
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
    return 0


def mlm_similarity(query, table):
    ''' Language modeling score between query and multi-field document repr. of the table '''
    return 0
