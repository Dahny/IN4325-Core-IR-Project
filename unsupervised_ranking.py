from extraction import read_tables, INPUT_FILE_TABLES, read_queries, INPUT_FILE_QUERIES
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from preprocessing.utils import preprocess_string
from nltk.corpus import stopwords
import rank_bm25
import pandas as pd
import json
import numpy as np
import nltk
nltk.download('stopwords')


# Either single-field or multi-field
def generate_features(method='single-field'):
    tables_raw = read_tables(INPUT_FILE_TABLES)
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
    # Preprocess the initial string
    field_result = preprocess_string(field_result)
    filtered_result = []
    stop_words = set(stopwords.words('english'))
    for w in field_result:
        if w not in stop_words:
            filtered_result.append(w)

    return filtered_result


def init_elasticsearch(host='localhost', port=9200):
    return Elasticsearch([{'host': host, 'port': port}])


def elasticsearch_index_single_field(es, features):
    counter = 0
    for (table, data) in features["data"].iteritems():
        field = {
                "data": data
                }
        es.create(index=table, id=counter, body=field)
        counter += 1


def gendata(features, method='single-field'):
    for (table, row) in features.iterrows():
        fields = {}
        for col, data in row.iteritems():
            fields[col] = data

        yield {
            "_index": method,
            "_type": "_doc",
            "_id": table,
            method: fields
        }


def bulk_index(es, features, method='single-field'):
    # index with bulk loads of 200 because of timeout issue
    i = 0
    for j in range(100, 3000, 100):
        print('indexing table number:', j)
        helpers.bulk(es, gendata(features[i:j], method), request_timeout=2000)
        i = j


def create_index_similarity(es, features, method='single-field'):
    mu = get_avg_length_docs(features)

    fields = {}

    # All of the fields will be a list of keywords
    field_type = {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                }

    # Take example row to generate data type body
    example = features.iloc[1]
    for col, _ in example.iteritems():
        fields[col] = field_type

    es.indices.create(index=method, body={
        "mappings": {
            "properties": {
                method: {
                    "properties": fields
                }
            }
        },
        "settings": {
            "similarity": {
                "default": {
                    "type": "LMDirichlet",
                    "mu": mu
                }
            }
        }
    })



def search_query_single_field(es, query, index='single-field', size=10, search_type="dfs_query_then_fetch", explain=False):
    return es.search(body={
                    'query': {
                        'match': {
                            "single-field.data": query
                        }
                    }
    }, index=index, size=size, search_type=search_type, request_timeout=3000, explain=explain)


def search_query_multi_field(es, query, index='multi-field', size=10, search_type="dfs_query_then_fetch", agg_type="most_fields",
                             explain=False):
    return es.search(body={
                        'query': {
                            'multi_match': {
                                "query": query,
                                "type": agg_type,
                                "fields": ["multi-field.caption", "multi-field.pgTitle", "multi-field.secondTitle",
                                           "multi-field.table_headings", "multi-field.table_body"]

                            }
                        }
    }, index=index, size=size, search_type=search_type, request_timeout=3000, explain=explain)


# Get avg doc length for mu
def get_avg_length_docs(features):
    lengths = {}
    for table, row in features.iterrows():
        length = 0
        for _, data in row.iteritems():
            length += len(data)
        lengths[table] = length
    mean = np.mean(list(lengths.values()))
    return mean


# This method only works when updating the parameter of the similarity
# Not the similarity itself
def change_similarity_Dirichlet(es, mu=2000, index="_all"):
    # First close all indices in order to update th settings
    es.indices.close(index=index, request_timeout=8000)

    # Update settings of all the indices
    es.indices.put_settings(index=index, body={
        "settings": {
            "index": {
                "similarity": {
                    "default": {
                        "type": "LMDirichlet",
                        "mu": mu
                    }
                }
            }
        }
    }, request_timeout=8000)

    # Reopen all indices
    es.indices.open(index=index, request_timeout=8000)


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
            query_results[item['_id']] = item['_score']

        scores[query] = query_results

    file = open(name, 'w')
    file.write(json.dumps(scores, indent=2))
    file.close()


def pipeline(host='localhost', port=9200):
    print("Starting pipeline...")

    print("Init connection with ElasticSearch")
    es = init_elasticsearch(host, port)

    # Extract text from tables
    print("Extracting Features")
    s_features = generate_features(es, 'single-field')
    m_features = generate_features(es, 'multi-field')

    # Create indices
    print("Creating indices")
    create_index_similarity(es, s_features, 'single-field')
    create_index_similarity(es, m_features, 'multi-field')

    # Insert document data into the indices
    print("inserting data into indices")
    bulk_index(es, s_features, 'single-field')
    bulk_index(es, m_features, 'multi-field')

    # Get rankings from queries and write to file
    print("writing rankings to file")
    queries = read_queries(INPUT_FILE_QUERIES)
    get_all_scores(es, queries, 'single-field', 'single-field-rankings.json')
    get_all_scores(es, queries, 'multi-field', 'multi-field-rankings.json')

    print("Pipeline done!")
