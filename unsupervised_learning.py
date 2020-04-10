import extraction
import rank_bm25
import pandas as pd
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.util import ngrams
from nltk.lm import NgramCounter
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import re
import string

# Either single-field or multi-field
def generate_features(method = 'single-field'):
    tables_raw = extraction.read_tables(extraction.INPUT_FILE_TABLES)
    tables = pd.DataFrame.from_dict(tables_raw)
    output = pd.DataFrame()

    if method == 'single-field':
        output = pd.DataFrame(columns=['table', 'data'])
    if method == 'multi-field':
        output = pd.DataFrame(columns=['table', 'pgTitle', 'secondTitle', 'caption', 'table_headings', 'table_body'])

    for table, fields in tables.iteritems():
        pgTitle = tables[table]['pgTitle']
        secondTitle = tables[table]['secondTitle']
        caption = tables[table]['caption']
        table_headings = tables[table]['title']
        table_body = tables[table]['data']

        if method == 'single-field':
            data = single_field_representation(table, pgTitle, secondTitle, caption, table_headings, table_body)
        elif method == 'multi-field':
            data = multi_field_representation(table, pgTitle, secondTitle, caption, table_headings, table_body)
        else:
            # This should never happen
            raise Exception("No document representation method is chosen")
        output = output.append(data, ignore_index=True)
    output.set_index('table', inplace=True)
    return output

def single_field_representation(table, pgTitle, secondTitle, caption, table_headings, table_body):
    # collect all textual data and put it into 1 list
    data = [pgTitle, secondTitle, caption] + table_headings
    for row in table_body:
        data = data + row

    # preprocess the text
    preprocessed_data = []
    for item in data:
        preprocessed_data = preprocessed_data + preprocess_field(item)

    return {'table': table, 'data': preprocessed_data}


def multi_field_representation(table, pgTitle, secondTitle, caption, table_headings, table_body):
    # store and preprocess all data into seperate fields
    pp_pgTitle = preprocess_field(pgTitle)
    pp_secondTitle = preprocess_field(secondTitle)
    pp_caption = preprocess_field(caption)

    pp_table_headings = []
    for header in table_headings:
        pp_table_headings = pp_table_headings + preprocess_field(header)

    pp_table_body = []
    for row in table_body:
        for col in row:
            pp_table_body = pp_table_body + preprocess_field(col)

    return {'table': table, 'pgTitle': pp_pgTitle, 'secondTitle': pp_secondTitle, 'caption': pp_caption,
            'table_headings': pp_table_headings, 'table_body': pp_table_body}



# def n_gram_language_model_single_field(features, n = 1):
#     n = 1
#     train_data, padded_sents = padded_everygram_pipeline(n, features['data'])
#
#     model = MLE(n)  # Lets train a 3-grams maximum likelihood estimation model.
#     model.fit(train_data, padded_sents)
#
#     queries = extraction.read_queries(extraction.INPUT_FILE_QUERIES)
#     queries = pd.read_csv('data/queries.txt', sep=" ", header=2, names=['number', 'query'])
#
#     text_unigrams = [ngrams(sent, n) for sent in features["data"]]
#     ngram_counts = NgramCounter(text_unigrams)


def bm25_ranking_single_field(features, query):
    bm25 = rank_bm25.BM25Okapi(features['data'])
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)

    ranking_results = features
    ranking_results['score'] = doc_scores
    ranking_results = ranking_results.sort_values(by='score')
    return ranking_results


def preprocess_field(field):
    field_result = field
    # lowercase
    field_result = field_result.lower()
    # removing numbers
    field_result = re.sub(r'\d', '', field_result)
    # removing punctuation
    field_result = field_result.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    # removing white spaces
    field_result = field_result.strip()
    # tokenize into words
    field_result = word_tokenize(field_result, language='english')
    # remove stop words
    filtered_result = []
    stop_words = set(stopwords.words('english'))
    for w in field_result:
        if w not in stop_words:
            filtered_result.append(w)

    return filtered_result



def init_elasticsearch(host = 'localhost', port = 9200):
    return Elasticsearch([{'host': host, 'port': port}])


def elasticsearch_index_single_field(features):
    es = init_elasticsearch()

    counter = 0
    for (table, data) in features["data"].iteritems():
        field = {
                "data": data
                }
        es.create(index=table, id=counter, body=field)
        counter += 1


def gendata_single_field(features):
    for (table, data) in features["data"].iteritems():
        yield {
                "_index": table,
                "_type": "_doc",
                "data": data,
                }

def gendata_multi_field(features):
    for (table, row) in features.iterrows():
        fields = {}
        for col, data in row.iteritems():
            fields[col] = data

        yield {
            "_index": table,
            "_type": "_doc",
            "multi-fields": fields
        }

def bulk_index(features, method='single-field'):
    es = init_elasticsearch()

    # index with bulk loads of 200 because of timeout issue
    i = 0
    for j in range(200, 3000, 200):
        if method == 'single-field':
            helpers.bulk(es, gendata_single_field(features[i:j]), request_timeout=2000)
        if method == 'multi-field':
            helpers.bulk(es, gendata_multi_field(features[i:j]), request_timeout=2000)
        i = j


def create_indices_similarity_single_field(features):
    es = init_elasticsearch()

    for (table, _) in features["data"].iteritems():
        es.indices.create(index=table, body={
            "mappings": {
                "properties": {
                    "data": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    }
                }
            },
            "settings": {
                "similarity": {
                    "default": {
                        "type": "LMDirichlet",
                        "mu": 2000
                    }
                }
            }
        })


def search_query_single_field(es, query, index=None, size=10, search_type="dfs_query_then_fetch", explain=False):
    return es.search(body={
                    'query': {
                        'match': {
                            "data": query
                        }
                    }
    }, index=index, size=size, search_type=search_type, request_timeout=3000, explain=explain)


def search_query_multi_field(es, query, index=None, size=10, search_type="dfs_query_then_fetch", agg_type="most_fields",
                             explain=False):
    return es.search(body={
                        'query': {
                            'multi_match': {
                                "query": query,
                                "type": agg_type,
                                "fields": ["multi-fields.caption", "multi-fields.pgTitle", "multi-fields.secondTitle",
                                           "multi-fields.table_headings", "multi-fields.table_body"]

                            }
                        }
    }, index=index, size=size, search_type=search_type, request_timeout=3000, explain=explain)

# This method doesn't work as apparently you can't update the similarity of indices
def change_similarity_Dirichlet(es, mu=2000):
    # First close all indices in order to update th settings
    es.indices.close(index="_all", request_timeout=8000)

    # Update settings of all the indices
    es.indices.put_settings(index="_all", body={
        "settings": {
            "index": {
                "similarity": {
                    "my_similarity": {
                        "type": "LMDirichlet",
                        "mu": mu
                    }
                }
            }
        }
    }, request_timeout=8000)

    # Reopen all indices
    es.indices.open(index="_all", request_timeout=8000)


def get_all_scores(es, queries, method='single-field', name='rankings.json'):
    scores = {}
    for query in queries.values():
        if method == 'single-field':
            results = search_query_single_field(es, query, size=3000)['hits']['hits']
        if method == 'multi-field':
            results = search_query_multi_field(es, query, size=3000)['hits']['hits']
        print('found', len(results), 'results for query:', query)

        query_results = {}
        for item in results:
            query_results[item['_index']] = item['_score']

        scores[query] = query_results

    file = open(name, 'w')
    file.write(json.dumps(scores, indent=2))
    file.close()
