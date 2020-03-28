import pandas as pd
import json
from features.baselineFeatures import compute_baseline_features
from features.semanticFeatures import compute_semantic_features


FINAL_HEADERS = ['query_id','query','table_id','row','col','nul',
  'in_link','out_link','pgcount','tImp','tPF','leftColhits','SecColhits',
  'bodyhits','PMI','qInPgTitle','qInTableTitle','yRank','csr_score','idf1',
  'idf2','idf3','idf4','idf5','idf6','max','sum','avg','sim','emax','esum','eavg',
'esim','cmax','csum','cavg','csim','remax','resum','reavg','resim','query_l','rel']

INPUT_FILE_QRELS = 'data/qrels.txt'
INPUT_FILE_QUERIES = 'data/queries.txt'
INPUT_FILE_TABLES = 'data/raw_table_data.json'

FEATURE_FILE = 'data/features.csv'


def read_qrels(input_file: str):
    data = pd.read_table(input_file, sep = '\t', names=['query_id', 'unknown', 'table_id', 'rel'])
    data = data.drop(columns=['unknown'])
    return data


def read_queries(input_file: str):
    queries = {}
    with open(input_file, 'r') as f:
        for line in f:
            split = line.split(' ')
            queries[int(split[0])] = ' '.join(split[1:]).strip()
    return queries


def read_tables(input_file: str):
    with open(input_file, 'r') as f:
        tables = json.loads(f.read())
    return tables


def feature_extraction(input_file_qrels: str, input_file_queries: str, input_file_tables: str, output_file: str):
    query_col = 'query'
    table_col = 'raw_table_data'
    
    data_table = read_qrels(input_file_qrels)
    queries = read_queries(input_file_queries)
    tables = read_tables(input_file_tables)

    data_table[query_col] = data_table['query_id'].map(queries)
    data_table[table_col] = data_table['table_id'].map(tables) 

    import time
    start_time = time.time()
    print('---------- START COMPUTING BASELINE FEATURES ----------')
    data_table = compute_baseline_features(data_table, query_col, table_col)
    mid_time = time.time()
    print(f'---------- TOOK {mid_time - start_time} SECONDS FOR BASELINE FEATURE EXTRACTION ----------')
    print('---------- START COMPUTING SEMANTIC FEATURES ----------')
    data_table = compute_semantic_features(data_table, query_col, table_col)
    print(f'---------- TOOK {time.time() - mid_time} SECONDS FOR SEMANTIC FEATURE EXTRACTION ----------')

    print("Computing all features completed")
    
    missing_features = set(FINAL_HEADERS) - set(data_table.columns.values)
    print(f'Missing the following features from their experiment:\n{missing_features}')

    # df.to_csv(output_file, sep=',', index=False, columns=FINAL_HEADERS)
    data_table.drop(columns=[table_col]).to_csv(output_file, sep=',', index=False)


if __name__ == '__main__':
    feature_extraction(INPUT_FILE_QRELS, INPUT_FILE_QUERIES, INPUT_FILE_TABLES, FEATURE_FILE)