import pandas as pd

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

def compute_semantic_features(data_table, query_col, table_col):
    ''' Compute semantic features introduced by the paper '''

    ## See 3.1 in the paper for the following section
    q_word_based_set = data_table[query_col].map(query_word_based_set)
    t_word_based_set = data_table[table_col].map(table_word_based_set)
    
    q_entity_based_set = data_table[query_col].map(query_entity_based_set)
    t_entity_based_set = data_table[table_col].map(table_entity_based_set)

    ## See 3.2 in the paper for the following section
    q_word_embeddings = q_word_based_set.map(query_word_embeddings)
    t_word_embeddings = t_word_based_set.map(table_word_embeddings)
    
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
            data_table[i[0] + 'sim'] = i[1].apply(lambda x: early_fusion_incl_tfidf(x.iloc[0], x.iloc[1]), axis=1)
        else:
            data_table[i[0] + 'sim'] = i[1].apply(lambda x: early_fusion(x.iloc[0], x.iloc[1]), axis=1)
        print(type(i[1].apply(lambda x: late_fusion(x.iloc[0], x.iloc[1]), axis=1)))
        data_table[i[0] + 'avg'], data_table[i[0] + 'max'], data_table[i[0] + 'sum'] = \
            zip(*i[1].apply(lambda x: late_fusion(x.iloc[0], x.iloc[1]), axis=1))

    return data_table


## Function related to 3.1 Content Extraction
def query_word_based_set(query):
    ''' All word tokens of the query in a set '''
    return {}


def table_word_based_set(table):
    ''' All word tokens from the title, captions and heading of the table '''
    return {}


def query_entity_based_set(query, k=10):
    ''' Get entities based on DBpedia knowledge base only return the top 'k' entitites '''
    return {}


def table_entity_based_set(table, k=10):
    ''' Get the entities using DBpedia of the core column + page title + table caption and return the top 'k' entities '''
    ''' The core colum is defined the column with the highest ratio of cells containing an entity '''
    return {}


## Functions related to 3.2 Semantic Representations
def query_word_embeddings(query):
    ''' Returns a set with the word embeddings, from word2vec, for each word. '''
    return {}


def table_word_embeddings(table):
    ''' Returns a set with the word embeddings, from word2vec, for each word. '''
    return {}


def query_entity_embeddings(query):
    ''' Returns a set with the graph embeddings, from RDF2vec, for each entity. '''
    return {}


def table_entity_embeddings(table):
    ''' Returns a set with the graph embeddings, from RDF2vec, for each entity. '''
    return {}


## Functions related to 3.3 Similarity Measures
def early_fusion(query, table):
    ''' Calculate the centroids for both query and table vectors and return similarity between them '''
    return 0


def early_fusion_incl_tfidf(query, table):
    ''' Calculate the centroids for both query and table vectors and return similarity between them the centroids are weighed by TFIDF'''
    return 0


def late_fusion(query, table):
    ''' Calculate the cosine similarity between all vector pairs between query and table and return the avg, max and sum '''
    return 0, 0, 0