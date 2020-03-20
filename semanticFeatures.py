import pandas as pd
import numpy as np
from nltk import word_tokenize

import gensim.downloader as api
from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity

import re
import os

# FEATURES from their features.csv file:
# {
#     'sim',
#     'max', 
#     'avg', 
#     'sum', 

#     'remax', 
#     'reavg',
#     'resum', 
#     'resim', 

#     'eavg', 
#     'emax', 
#     'esum', 
#     'esim', 

#     'csim', 
#     'csum', 
#     'cavg', 
#     'cmax'
# }

def compute_semantic_features(data_table, query_col='query', table_col='raw_table_data'):
    ''' Compute semantic features introduced by the paper '''

    ## See 3.1 in the paper for the following section
    q_word_based_set = data_table[query_col].map(query_word_based_set)
    t_word_based_set = data_table[table_col].map(table_word_based_set)
    
    q_entity_based_set = data_table[query_col].map(query_entity_based_set)
    t_entity_based_set = data_table[table_col].map(table_entity_based_set)

    ## See 3.2 in the paper for the following section
    import time
    start_time = time.time()
    wv_file_name = 'word2vec.kv'
    if not os.path.exists(wv_file_name):
        print('---------- DID NOT FIND Word2Vec, START DOWNLOADING ----------')
        wv = api.load('word2vec-google-news-300')
        wv = wv.wv
        wv.save(wv_file_name)
    else:        
        print('---------- FOUND Word2Vec, START READING ----------')
        wv = KeyedVectors.load(wv_file_name, mmap='r')
    print(f'---------- TOOK {time.time() - start_time} SECONDS TO LOAD WORD2VEC ----------')
    q_word_embeddings = q_word_based_set.apply(lambda x: word_embeddings(x, wv))
    t_word_embeddings = t_word_based_set.apply(lambda x: word_embeddings(x, wv))
    # Free memory from word2vec
    wv = 0
    
    q_entity_embeddings = q_entity_based_set.map(query_entity_embeddings)
    t_entity_embeddings = t_entity_based_set.map(table_entity_embeddings)

    # TODO: Bag-of-entities + Bag-of-categories

    ## See 3.3 in the paper for the following section
    # I believe no prefix represents word embeddings, 're' is graph embeddings, 'c' bag of category and 'e' bag of entity
    to_compare = [
        ['', pd.concat([q_word_embeddings, t_word_embeddings], axis=1)], 
        # ['re', pd.concat([q_entity_embeddings, table_entity_embeddings], axis=1)]
    ]
    for i in to_compare:
        if i[0] == '':
            # TODO: Change this to weighed TF_IDF version
            data_table[i[0] + 'sim'] = i[1].apply(lambda x: early_fusion(x.iloc[0], x.iloc[1]), axis=1)
        else:
            data_table[i[0] + 'sim'] = i[1].apply(lambda x: early_fusion(x.iloc[0], x.iloc[1]), axis=1)
        data_table[i[0] + 'avg'], data_table[i[0] + 'max'], data_table[i[0] + 'sum'] = \
            zip(*i[1].apply(lambda x: late_fusion(x.iloc[0], x.iloc[1]), axis=1))

    return data_table


## Function related to 3.1 Content Extraction
def query_word_based_set(query):
    ''' All word tokens of the query in a set '''
    return word_tokenize(query)


def table_word_based_set(table):
    ''' All word tokens from the title, captions and heading of the table '''
    pgTable_tokens = word_tokenize(table['pgTitle'])
    caption_tokens = word_tokenize(table['caption'])
    headers_tokens = process_headers(table['title'])

    result = [x.lower() for x in list(set(pgTable_tokens + caption_tokens + headers_tokens))]
    return result


def query_entity_based_set(query, k=10):
    ''' Get entities based on DBpedia knowledge base only return the top 'k' entitites '''
    return {}


def table_entity_based_set(table, k=10):
    ''' Get the entities using DBpedia of the core column + page title + table caption and return the top 'k' entities '''
    ''' The core colum is defined the column with the highest ratio of cells containing an entity '''
    return {}


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


def query_entity_embeddings(query):
    ''' Returns a set with the graph embeddings, from RDF2vec, for each entity. '''
    return {}


def table_entity_embeddings(table):
    ''' Returns a set with the graph embeddings, from RDF2vec, for each entity. '''
    return {}


## Functions related to 3.3 Similarity Measures
def early_fusion(query, table):
    ''' Calculate the centroids for both query and table vectors and return similarity between them '''
    average_query = query.mean(axis=0)
    average_table = table.mean(axis=0)
    return cosine_similarity(average_query.reshape(1, -1), average_table.reshape(1, -1))[0][0]


def early_fusion_incl_tfidf(query, table):
    ''' Calculate the centroids for both query and table vectors and return similarity between them the centroids are weighed by TFIDF'''
    return 0


def late_fusion(query, table):
    ''' Calculate the cosine similarity between all vector pairs between query and table and return the avg, max and sum '''
    all_pairs = np.zeros(len(query) * len(table))
    for i, q in enumerate(query):
        for j, t in enumerate(table):
            all_pairs[i*j] = cosine_similarity(q.reshape(1, -1) ,t.reshape(1, -1))
    return all_pairs.mean(), all_pairs.max(), all_pairs.sum()


## Helper functions
def process_headers(headers):
    headers = [x.strip() for x in word_tokenize(re.sub(r'-|/|\||\_|\[|\]|<[^>]*>', ' ', ' '.join(headers)))]
    return headers