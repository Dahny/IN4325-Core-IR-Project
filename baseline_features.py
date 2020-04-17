# FINAL_HEADERS = ['query_id','query','table_id','row','col','nul',
#   'in_link','out_link','pgcount','tImp','tPF','leftColhits','SecColhits',
#   'bodyhits','PMI','qInPgTitle','qInTableTitle','yRank','csr_score','idf1',
#   'idf2','idf3','idf4','idf5','idf6','max','sum','avg','sim','emax','esum','eavg',
#   'esim','cmax','csum','cavg','csim','remax','resum','reavg','resim','query_l','rel']

import json
import math
from nltk import word_tokenize
from preprocessing.utils import preprocess_string, get_entity_to_information_dict, read_json

# # Load lookup dictionaries
# with open('dictionaries/words_page_titles.json') as f:
#     dict_page_titles = json.load(f)
#     f.close()
# with open('dictionaries/words_section_titles.json') as f:
#     dict_section_titles = json.load(f)
#     f.close()
# with open('dictionaries/words_captions.json') as f:
#     dict_captions = json.load(f)
#     f.close()
# with open('dictionaries/words_headers.json') as f:
#     dict_headers = json.load(f)
#     f.close()
# with open('dictionaries/words_data.json') as f:
#     dict_data = json.load(f)
#     f.close()
# with open('dictionaries/wikipages_per_query.json') as f:
#     dict_query_wikipages = json.load(f)
#     f.close()
# with open('data/multi_field_rankings.json') as f:
#     rankings = json.load(f)
#     f.close()

print('----- START READING INFORMATION FILE -----')
# dict_information = get_entity_to_information_dict('dictionaries/entities_to_information.csv')
print('----- FINISHED READING INFORMATION FILE -----')

# Total number of documents
n_documents = 2932


def compute_baseline_features(data_table, query_col='query', table_col='raw_table_data'):
    """
    Compute all features regarded as baseline features in the paper
    :param data_table:
    :param query_col:
    :param table_col:
    :return:
    """

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
        data_table[k] = data_table[query_col].map(v)

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
        'pgcount': average_page_view,
    }
    for k, v in table_features.items():
        data_table[k] = data_table[table_col].map(v)

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
        if k == 'csr_score':
            data_table[k] = data_table.apply(lambda x: v(x[query_col], x['table_id']), axis=1)
        else:
            data_table[k] = data_table.apply(lambda x: v(x[query_col], x[table_col]), axis=1)

    # Free memory?
    dict_information = 0

    return data_table


# Query featuresCreation

def query_length(query):
    """
    Takes the query and returns the length
    :param query:
    :return:
    """
    return len(query.split(' '))


def idf_page_title(query):
    """
    Takes the query and returns the sum of the IDF scores of the words in the page titles
    :param query:
    :return:
    """
    preprocessed_query = preprocess_string(query)
    final_idf = 0
    for term in preprocessed_query:
        if term in dict_page_titles:
            final_idf += compute_idf_t(n_documents, len(dict_page_titles[term]))
    return final_idf


def idf_section_title(query):
    """
    Takes the query and returns the sum of the IDF scores of the words in the section titles
    :param query:
    :return:
    """
    preprocessed_query = preprocess_string(query)
    final_idf = 0
    for term in preprocessed_query:
        if term in dict_section_titles:
            final_idf += compute_idf_t(n_documents, len(dict_section_titles[term]))
    return final_idf


def idf_table_caption(query):
    """
    Takes the query and returns the sum of the IDF scores of the words in the table captions
    :param query:
    :return:
    """
    preprocessed_query = preprocess_string(query)
    final_idf = 0
    for term in preprocessed_query:
        if term in dict_captions:
            final_idf += compute_idf_t(n_documents, len(dict_captions[term]))
    return final_idf


def idf_table_heading(query):
    """
    Takes the query and returns the sum of the IDF scores of the words in the table headings
    :param query:
    :return:
    """
    preprocessed_query = preprocess_string(query)
    final_idf = 0
    for term in preprocessed_query:
        if term in dict_headers:
            final_idf += compute_idf_t(n_documents, len(dict_headers[term]))
    return final_idf


def idf_table_body(query):
    """
    Takes the query and returns the sum of the IDF scores of the words in the table bodies
    :param query:
    :return:
    """
    preprocessed_query = preprocess_string(query)
    final_idf = 0
    for term in preprocessed_query:
        if term in dict_data:
            final_idf += compute_idf_t(n_documents, len(dict_data[term]))
    return final_idf


def idf_catch_all(query):
    """
    Takes the query and returns the sum of the IDF scores of the words in the all text of the tables
    :param query:
    :return:
    """
    preprocessed_query = preprocess_string(query)
    final_idf = 0
    for term in preprocessed_query:
        if term in dict_page_titles:
            final_idf += compute_idf_t(n_documents, len(dict_page_titles[term]))
        if term in dict_section_titles:
            final_idf += compute_idf_t(n_documents, len(dict_section_titles[term]))
        if term in dict_captions:
            final_idf += compute_idf_t(n_documents, len(dict_captions[term]))
        if term in dict_headers:
            final_idf += compute_idf_t(n_documents, len(dict_headers[term]))
        if term in dict_data:
            final_idf += compute_idf_t(n_documents, len(dict_data[term]))
    return final_idf


def compute_idf_t(n, dft):
    """
    Compute the
    :param n:
    :param dft:
    :return:
    """
    return math.log(n / dft)


# Table featuresCreation

def number_of_rows(table):
    """
    Takes the table and returns the number of rows
    :param table:
    :return:
    """
    return len(table['data'])


def number_of_columns(table):
    """
    Takes the table and return the number of columns
    :param table:
    :return:
    """
    if len(table['data']) > 0:
        return len(table['data'][0])
    return 0


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
    title = table['pgTitle'].replace(' ', '_')
    if title not in dict_information:
        return 0
    else:
        return len(dict_information[title]['inlinks'])


def number_of_out_links(table):
    """
    Takes the table and returns the number of outlinks to the page embedding the table
    :param table:
    :return:
    """
    title = table['pgTitle'].replace(' ', '_')
    if title not in dict_information:
        return 0
    else:
        return len(dict_information[title]['outlinks'])


def table_importance(table):
    """
    Takes the table and returns the inverse of the number of tables on the page
    :param table:
    :return:
    """
    title = table['pgTitle'].replace(' ', '_')
    if title not in dict_information:
        return 0
    else:
        nr_of_tables = float(dict_information[title]['nr_of_tables'])
        if nr_of_tables == 0.0:
            return 0.0
        return 1.0 / float(dict_information[title]['nr_of_tables'])


def page_fraction(table):
    """
    Takes the table and returns the inverse of the number of tables on the page
    :param table:
    :return:
    """
    title = table['pgTitle'].replace(' ', '_')
    if title not in dict_information:
        return 0
    else:
        table_size = number_of_rows(table) * number_of_columns(table)
        nr_of_words = float(dict_information[title]['nr_of_words'])
        if nr_of_words == 0.0:
            return 0.0
        return table_size / nr_of_words


def pmi(table):
    """
    Takes the table and returns the ACSDb-based schema coherency score
    :param table:
    :return:
    """
    average_pmi = 0
    counter = 0
    preprocessed_headers = list(map(lambda x: preprocess_string(x), table['title']))
    for i in range(len(preprocessed_headers) - 1):
        for j in range(i + 1, len(preprocessed_headers)):
            counter += 1
            pmi = 0
            for h1 in preprocessed_headers[i]:
                for h2 in preprocessed_headers[j]:
                    pmi += compute_pmi(h1, h2, n_documents, dict_headers)
            if pmi == 0:
                average_pmi = 0
            else:
                average_pmi += (pmi / (len(preprocessed_headers[i]) * len(preprocessed_headers[j])))
    if counter == 0:
        return 0.0
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
    if p_a_b == 0.0:
        return 0.0
    return math.log(p_a_b / (p_a * p_b))


def average_page_view(table):
    """
    Takes the table and returns the average page views of the Wikipedia page over a given time interval
    :param table:
    :return:
    """
    title = table['pgTitle'].replace(' ', '_')
    if title not in dict_information:
        return 0.0
    else:
        return float(dict_information[title]['page_views'])


# Query-table featuresCreation

def term_frequency_query_in_left_column(query, table):
    """
    Total query term frequency in the leftmost column cells
    :param query:
    :param table:
    :return:
    """
    if len(table['data']) > 0:
        tokenized_query = word_tokenize(query.lower())
        first_column = [i[0] for i in table['data']]
        tokenized_first_column = list(map(lambda x: preprocess_string(x), first_column))
        number_found = 0
        for query_token in tokenized_query:
            for cell in tokenized_first_column:
                if query_token in cell:
                    number_found += 1

        return number_found
    return -1


def term_frequency_query_in_second_column(query, table):
    """
    Total query term frequency in second-to-leftmost column cells
    :param query:
    :param table:
    :return:
    """
    if len(table['data']) > 0:
        if len(table['data'][0]) > 1:
            tokenized_query = word_tokenize(query.lower())
            second_column = [i[1] for i in table['data']]
            tokenized_second_column = list(map(lambda x: preprocess_string(x), second_column))
            number_found = 0
            for query_token in tokenized_query:
                for cell in tokenized_second_column:
                    if query_token in cell:
                        number_found += 1
            return number_found
        return -1
    return -1


def term_frequency_query_in_table_body(query, table):
    """
    Total query term frequency in the table body
    :param query:
    :param table:
    :return:
    """
    if len(table['data']) > 0:

        tokenized_query = word_tokenize(query.lower())
        number_found = 0

        for row in table['data']:
            data_row = list(map(lambda x: preprocess_string(x), row))
            for cell in data_row:
                for query_token in tokenized_query:
                    if query_token in cell:
                        number_found += 1
        return number_found
    return -1


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
    wikipages = dict_query_wikipages[query]
    for i in range(len(wikipages)):
        if wikipages[i] == table['pgTitle']:
            return i + 1
    return max_result + 1


def mlm_similarity(query, table_id):
    """
    Language modeling score between query and multi-field document repr. of the table
    :param query:
    :param table:
    :return:
    """
    # Return the scores based on the query
    scores_query = rankings[query]
    score = 0

    # return the score if there is one, else return 0
    if table_id in scores_query:
        score = scores_query[table_id]

    return score
