import extraction
import rank_bm25
import pandas as pd
from nltk.util import ngrams
from nltk.lm import NgramCounter
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

# Either single-field or multi-field
def generate_features(method = 'single-field'):
    tables_raw = extraction.read_tables(extraction.INPUT_FILE_TABLES)
    tables = pd.DataFrame.from_dict(tables_raw)
    output = pd.DataFrame(columns=['table', 'data'])
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
            break
        output = output.append(data, ignore_index=True)
    output.set_index('table', inplace=True)
    return output


def single_field_representation(table, pgTitle, secondTitle, caption, table_headings, table_body):
    # collect all textual data and put it into 1 list
    data = [pgTitle, secondTitle, caption] + table_headings
    for row in table_body:
        data = data + row
    return {'table': table, 'data': data}


def multi_field_representation(table, pgTitle, secondTitle, caption, table_headings, table_body):
    return {'table': table, 'pgTitle': pgTitle, 'secondTitle': secondTitle, 'caption': caption,
            'table_headings': table_headings, 'table_body': table_body}



def n_gram_language_model_single_field(features, n = 1):
    n = 1
    train_data, padded_sents = padded_everygram_pipeline(n, features['data'])

    model = MLE(n)  # Lets train a 3-grams maximum likelihood estimation model.
    model.fit(train_data, padded_sents)

    extraction.read_queries(extraction.INPUT_FILE_QUERIES)
    queries = pd.read_csv('data/queries.txt', sep=" ", header=2, names=['number', 'query'])

    text_unigrams = [ngrams(sent, n) for sent in features["data"]]
    ngram_counts = NgramCounter(text_unigrams)

def bm25_ranking_single_field(features, query):
    bm25 = rank_bm25.BM25Okapi(features['data'])
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)

    ranking_results = features
    ranking_results['score'] = doc_scores
    ranking_results = ranking_results.sort_values(by='score')
    return ranking_results


