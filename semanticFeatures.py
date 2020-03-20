import pandas as pd
import numpy as np

# Word tokenizer
from nltk import word_tokenize

# Entity extraction
from SPARQLWrapper import SPARQLWrapper, JSON

# Word2vec
import gensim.downloader as api
from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity

import re
import os
import time
import requests


def compute_semantic_features(data_table, query_col='query', table_col='raw_table_data'):
    ''' Compute semantic features introduced by the paper '''

    ## See 3.1 in the paper for the following section
    words_in_query = data_table[query_col].map(tokenize_query)
    words_in_table = data_table[table_col].map(tokenize_table)
    
    entities_in_query = unique_query_execution(words_in_query, entities_from_words)

    # TODO: Implement entities from table function as accoreding to paper
    entities_in_table = words_in_table.map(entities_from_table)

    ## See 3.2 in the paper for the following section
    start_time = time.time()
    wv = load_word2vec('word2vec.kv')
    print(f'---------- TOOK {time.time() - start_time} SECONDS TO LOAD WORD2VEC ----------')
    q_word_embeddings = unique_query_execution(words_in_query, word_embeddings, wv)
    t_word_embeddings = words_in_table.apply(lambda x: word_embeddings(x, wv))
    # Free memory from word2vec
    wv = 0
    
    # TODO: Find some way to get entity embeddings
    q_entity_embeddings = entities_in_query.map(entity_embeddings)
    t_entity_embeddings = entities_in_table.map(entity_embeddings)

    # TODO: Bag-of-entities + Bag-of-categories

    ## See 3.3 in the paper for the following section
    # I believe no prefix represents word embeddings, 're' is graph embeddings, 'c' bag of category and 'e' bag of entity
    to_compare = [
        ['', pd.concat([q_word_embeddings, t_word_embeddings], axis=1)], 
        # ['re', pd.concat([q_entity_embeddings, table_entity_embeddings], axis=1)]
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
        headers_tokens = process_headers(table['title'])
    else:
        headers_tokens = []

    result = [x.lower() for x in list(set(pgTable_tokens + caption_tokens + headers_tokens))]
    return result

def entities_from_words(words, use_n_grams=True, k=10):
    ''' Return all words in the data that are entities '''
    # print("words " + str(len(words)))
    if use_n_grams:
        words_incl_n_grams = []
        for N in range(1, len(words) + 1):
            words_incl_n_grams += [' '.join(words[i:i+N]) for i in range(len(words)-N+1)]
        words = words_incl_n_grams
            
    result = [i for i in words if check_for_entity_in_dbpedia(i)]
    # print("results " + str(len(result)))
    return result


def entities_from_table(table, k=10):
    ''' Get entities based on DBpedia knowledge base only return the top 'k' entitites ''' 
    ''' Should also check for "core column" as described in the paper '''
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


def entity_embeddings(entities):
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
            all_pairs[i*(j+1)+j] = cosine_similarity(q.reshape(1, -1) ,t.reshape(1, -1))
    return all_pairs.mean(), all_pairs.max(), all_pairs.sum()


## Helper functions
def process_headers(headers):
    ''' Cleans up the text of the headers (this is related to entities in the headers, might find cleaner fix later) '''
    headers = [x.strip() for x in word_tokenize(re.sub(r'-|/|\||\_|\[|\]|<[^>]*>', ' ', ' '.join(headers)))]
    return headers


def check_for_entity_in_dbpedia(entity):
    ''' Checks the DBPedia API whether an entity exists with the name '''
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    sparql.setQuery("""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?uri ?label
    WHERE { 
        { ?uri rdfs:label \"""" + entity.lower() + """" } UNION
        { ?uri rdfs:label \"""" + entity.capitalize() + """" }.
        ?uri rdfs:label ?label
    }
    LIMIT 1
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    if len(results["results"]["bindings"]) > 0:
        return True
    else:
        return False


def load_word2vec(wv_file_name):
    ''' Gets the word2vec model traind on google news '''
    if not os.path.exists(wv_file_name):
        print('---------- DID NOT FIND Word2Vec, START DOWNLOADING ----------')
        wv = api.load('word2vec-google-news-300')
        wv = wv.wv
        wv.save(wv_file_name)
    else:        
        print('---------- FOUND Word2Vec LOCALLY, START READING ----------')
        wv = KeyedVectors.load(wv_file_name, mmap='r')
    return wv

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