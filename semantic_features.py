import pandas as pd
import numpy as np

# Word tokenizer
from nltk import word_tokenize

# Word2vec
import gensim.downloader as api
from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity

import re
import os
import time
import requests
import json

from preprocessing.utils import list_to_ngrams, read_json, preprocess_string, read_one_hot_encoding


def compute_semantic_features(data_table, query_col='query', table_col='raw_table_data'):
    ''' Compute semantic features introduced by the paper '''

    ## See 3.1 in the paper for the following section
    words_in_query = data_table[query_col].map(tokenize_query)
    words_in_table = data_table[table_col].map(tokenize_table)
    
    query_to_entities = read_json('dictionaries/query_to_entities.json')
    table_to_entities = read_json('dictionaries/table_to_entities.json')
    entities_in_query = data_table['query_id'].map(lambda x: query_to_entities[str(x)])
    entities_in_table = data_table['table_id'].map(lambda x: table_to_entities[x])

    query_entities_zero = 0
    table_entities_zero = 0
    for i in range(len(entities_in_query)):
        if len(entities_in_query[i]) == 0:
            query_entities_zero += 1
        if len(entities_in_table[i]) == 0:
            table_entities_zero += 1
    print(f'Query zero entities: {query_entities_zero}')
    print(f'Table zero entities: {table_entities_zero}')

    ## See 3.2 in the paper for the following section
    start_time = time.time()
    wv = load_word2vec('dictionaries/word2vec.kv')
    print(f'----- TOOK {time.time() - start_time} SECONDS TO LOAD WORD2VEC -----')
    q_word_embeddings = unique_query_execution(words_in_query, word_embeddings, wv)
    t_word_embeddings = words_in_table.apply(lambda x: word_embeddings(x, wv))
    # Free memory from word2vec
    wv = 0
    
    start_time = time.time()
    rdf2vec = load_rdf2vec('dictionaries/rdf2vec.json')
    print(f'----- TOOK {time.time() - start_time} SECONDS TO LOAD RDF2VEC -----')
    q_entity_embeddings = entities_in_query.apply(lambda x: entity_embeddings(x, rdf2vec))
    t_entity_embeddings = entities_in_table.apply(lambda x: entity_embeddings(x, rdf2vec))
    # Free memory from rdf2vec
    rdf2vec = 0

    query_entities_zero = 0
    table_entities_zero = 0
    for i in range(len(q_entity_embeddings)):
        if len(q_entity_embeddings[i]) == 0:
            query_entities_zero += 1
        if len(t_entity_embeddings[i]) == 0:
            table_entities_zero += 1
    print(f'Query zero rdf2vec: {query_entities_zero}')
    print(f'Table zero rdf2vec: {table_entities_zero}')

    # TODO: Bag-of-entities + Bag-of-categories
    print('---- Load one-hot encodings')
    entity_to_categories = read_one_hot_encoding('dictionaries/category_indices.json')
    entity_to_links = read_one_hot_encoding('dictionaries/link_indices.json')
    print('---- Done loading one-hot encodings')

    q_bag_of_entities = entities_in_query.apply(lambda x: get_one_hot_encodings(x, entity_to_links))
    t_bag_of_entities = entities_in_table.apply(lambda x: get_one_hot_encodings(x, entity_to_links))

    q_bag_of_categories = entities_in_query.apply(lambda x: get_one_hot_encodings(x, entity_to_categories))
    t_bag_of_categories = entities_in_table.apply(lambda x: get_one_hot_encodings(x, entity_to_categories))

    entities_to_links = 0
    entity_to_categories = 0

    ## See 3.3 in the paper for the following section
    # I believe no prefix represents word embeddings, 're' is graph embeddings, 'c' bag of category and 'e' bag of entity
    to_compare = [
        ['', pd.concat([q_word_embeddings, t_word_embeddings], axis=1)], 
        ['re', pd.concat([q_entity_embeddings, t_entity_embeddings], axis=1)],
        ['c', pd.concat([q_bag_of_categories, t_bag_of_categories], axis=1)],
        ['e', pd.concat([q_bag_of_entities, t_bag_of_entities], axis=1)]
    ]
    for i in to_compare:
        if i[0] == '':
            # TODO: Change this to weighed TF_IDF version (this is only for word_embeddings as per prefix)
            data_table[i[0] + 'sim'] = i[1].apply(lambda x: early_fusion(x.iloc[0], x.iloc[1]), axis=1)
        else:
            data_table[i[0] + 'sim'] = i[1].apply(lambda x: early_fusion(x.iloc[0], x.iloc[1]), axis=1)
        data_table[i[0] + 'avg'], data_table[i[0] + 'max'], data_table[i[0] + 'sum'] = \
            zip(*i[1].apply(lambda x: late_fusion(x.iloc[0], x.iloc[1]), axis=1))

    return data_table


## Function related to 3.1 Content Extraction
def tokenize_query(query):
    ''' All word tokens of the query in a set '''
    return word_tokenize(query)


def tokenize_table(table, incl_headers=True):
    ''' All word tokens from the title, captions and heading of the table '''
    pgTable_tokens = word_tokenize(table['pgTitle'])
    caption_tokens = word_tokenize(table['caption'])
    if incl_headers:
        headers_tokens = [x for title in table['title'] for x in preprocess_string(title)]
    else:
        headers_tokens = []

    result = [x.lower() for x in list(set(pgTable_tokens + caption_tokens + headers_tokens))]
    return result


## Functions related to 3.2 Semantic Representations
def word_embeddings(words, wv):
    ''' Returns a set with the word embeddings, from word2vec, for each word. '''
    result = []
    failed_words = []
    for word in words:
        try:
            result.append(wv[word])
        except KeyError:
            failed_words.append(word)

    # Print to debug in preprocessing of strings
    # print(f'Failed words:\n{failed_words}')

    return np.array(result)


def entity_embeddings(entities, rdf2vec):
    ''' Returns a set with the graph embeddings, from RDF2vec, for each entity. '''
    result = []
    failed_entities = []

    for entity in entities:
        try:
            result.append(rdf2vec[entity.lower()])
        except KeyError:
            failed_entities.append(entity)

    # # Print to debug
    # print(f'Failed entities:\n{failed_entities}')

    return np.array(result)


def get_one_hot_encodings(entities, dictionairy):
    result = []
    failed_entities = []

    for entity in entities:
        try:
            result.append(dictionairy[entity])
        except KeyError:
            failed_entities.append(entity)

    return np.array(result)


## Functions related to 3.3 Similarity Measures
def early_fusion(query, table):
    ''' Calculate the centroids for both query and table vectors and return similarity between them '''
    if len(query) == 0 or len(table) == 0:
        return -1
    average_query = query.mean(axis=0)
    average_table = table.mean(axis=0)
    return cosine_similarity(average_query.reshape(1, -1), average_table.reshape(1, -1))[0][0]


def early_fusion_incl_tfidf(query, table):
    ''' Calculate the centroids for both query and table vectors and return similarity between them the centroids are weighed by TFIDF'''
    return -1


def late_fusion(query, table):
    ''' Calculate the cosine similarity between all vector pairs between query and table and return the avg, max and sum '''
    if len(query) == 0 or len(table) == 0:
        return -1, -1, -1
    all_pairs = np.zeros(len(query) * len(table))
    for i, q in enumerate(query):
        for j, t in enumerate(table):
            all_pairs[i*len(table)-1+j] = cosine_similarity(q.reshape(1, -1) ,t.reshape(1, -1))
    return all_pairs.mean(), all_pairs.max(), all_pairs.sum()


### Helper functions
def load_word2vec(wv_file_name):
    ''' Gets the word2vec model traind on google news '''
    if not os.path.exists(wv_file_name):
        print('----- DID NOT FIND Word2Vec, START DOWNLOADING -----')
        wv = api.load('word2vec-google-news-300')
        wv = wv.wv
        wv.save(wv_file_name)
    else:        
        print('----- FOUND Word2Vec LOCALLY, START READING -----')
        wv = KeyedVectors.load(wv_file_name, mmap='r')
    return wv


def load_rdf2vec(path):
    if not os.path.exists(path):
        print('----- DID NOT FIND Rdf2Vec, DOWNLOAD FIRST -----')
    else:
        with open(path, 'r') as f:
            read_dict = json.loads(f.read())
        result = {}
        for k, v in read_dict.items():
            if k.lower() in result:
                # # Print to debug
                # print(f'Already found {k} in rdf2vec. Skipping this double.')
                pass
            else:
                result[k.lower()] = np.array(v)
        return result


def unique_query_execution(query_set, function, *args):
    ''' Makes sure the function is only ran once for each query but will return a full series '''
    current_input = 0
    current_ouput = 0
    result = np.zeros(len(query_set), dtype=pd.Series)
    for i, entry in enumerate(query_set):
        if not entry == current_input:
            current_output = function(entry, *args)
            current_input = entry
        result[i] = current_output
    return pd.Series(result)
