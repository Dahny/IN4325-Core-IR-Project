# FINAL_HEADERS = ['query_id','query','table_id','row','col','nul',
#   'in_link','out_link','pgcount','tImp','tPF','leftColhits','SecColhits',
#   'bodyhits','PMI','qInPgTitle','qInTableTitle','yRank','csr_score','idf1',
#   'idf2','idf3','idf4','idf5','idf6','max','sum','avg','sim','emax','esum','eavg',
#   'esim','cmax','csum','cavg','csim','remax','resum','reavg','resim','query_l','rel']

def compute_baseline_features(data_table, query_col='query', table_col='raw_table_data'):
    ''' Compute all features regarded as baseline features in the paper '''
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


### Query features
def query_length(query):
    ''' Takes the query and returns the length '''
    return len(query.split(' '))


def idf_page_title(query):
    ''' Takes the query and returns the sum of the IDF scores of the words in the page titles'''
    ''' ( IDF_t = log ( N / df_t ) 
        Here N is the number of documents and df_t the number of documents containing word t 
        This is then summed for all terms in the query ''' 
    ''' Note: this is also dependent on data about all documents (/tables). 
    We still need to get this here somehow '''
    return 0


def idf_section_title(query):
    ''' Takes the query and returns the sum of the IDF scores of the words in the section titles'''
    ''' ( IDF_t = log ( N / df_t ) 
        Here N is the number of documents and df_t the number of documents containing word t 
        This is then summed for all terms in the query ''' 
    ''' Note: this is also dependent on data about all documents (/tables). 
    We still need to get this here somehow '''
    return 0


def idf_table_caption(query):
    ''' Takes the query and returns the sum of the IDF scores of the words in the table captions '''
    ''' ( IDF_t = log ( N / df_t ) 
        Here N is the number of documents and df_t the number of documents containing word t 
        This is then summed for all terms in the query ''' 
    ''' Note: this is also dependent on data about all documents (/tables). 
    We still need to get this here somehow '''
    return 0


def idf_table_heading(query):
    ''' Takes the query and returns the sum of the IDF scores of the words in the table headings '''
    ''' ( IDF_t = log ( N / df_t ) 
        Here N is the number of documents and df_t the number of documents containing word t 
        This is then summed for all terms in the query ''' 
    ''' Note: this is also dependent on data about all documents (/tables). 
    We still need to get this here somehow '''
    return 0


def idf_table_body(query):
    ''' Takes the query and returns the sum of the IDF scores of the words in the table bodies '''
    ''' ( IDF_t = log ( N / df_t ) 
        Here N is the number of documents and df_t the number of documents containing word t 
        This is then summed for all terms in the query ''' 
    ''' Note: this is also dependent on data about all documents (/tables). 
    We still need to get this here somehow '''
    return 0


def idf_catch_all(query):
    ''' Takes the query and returns the sum of the IDF scores of the words in the all text of the tables '''
    ''' ( IDF_t = log ( N / df_t ) 
        Here N is the number of documents and df_t the number of documents containing word t 
        This is then summed for all terms in the query ''' 
    ''' Note: this is also dependent on data about all documents (/tables). 
    We still need to get this here somehow '''
    return 0


### Table features
def number_of_rows(table):
    ''' Takes the table and returns the number of rows '''
    return len(table['data']) + 1


def number_of_columns(table):
    ''' Takes the table and return the number of columns '''
    return len(table['title'])


def number_of_null(table):
    ''' Takes the table and returns the number of empty cells '''
    return 0


def number_of_in_links(table):
    ''' Takes the table and returns the number of inlinks to the page embedding the table '''
    return 0


def number_of_out_links(table):
    ''' Takes the table and returns the number of outlinks to the page embedding the table '''
    return 0


def table_importance(table):
    ''' Takes the table and returns the inverse of the number of tables on the page '''
    return 0


def page_fraction(table):
    ''' Takes the table and returns the ratio of table size to page size '''
    return 0


def pmi(table):
    ''' Takes the table and returns the ACSDb-based schema coherency score '''
    return 0


def page_views(table):
    ''' Takes the table and returns the page views of the table '''
    return 0


### Query-table features
def term_frequency_query_in_left_column(query, table):
    ''' Total query term frequency in the leftmost column cells '''
    return 0


def term_frequency_query_in_second_column(query, table):
    ''' Total query term frequency in second-to-leftmost column cells '''
    return 0


def term_frequency_query_in_table_body(query, table):
    ''' Total query term frequency in the table body '''
    return 0


def ratio_query_terms_in_page_title(query, table):
    '''  Ratio of the number of query tokens found in page title to total number of tokens '''
    return 0


def ratio_query_terms_in_table_title(query, table):
    ''' Ratio of the number of query tokens found in table title to total number of tokens '''
    return 0


def y_rank(query, table):
    '''  Rank of the table’s Wikipedia page in Web search engine results for the query '''
    return 0


def mlm_similarity(query, table):
    ''' Language modeling score between query and multi-field document repr. of the table '''
    return 0
