import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold

from evaluation_metrics import compute_NDCG, compute_ERR

def get_ndcg20_for_X_Y(X, y, groups, crossvalidation):
    for i, (train_index, test_index) in enumerate(crossvalidation.split(X, y, groups)):
        print(f'------------ RUNNING MODEL FOR SPLIT {i + 1} / {crossvalidation.get_n_splits()} ------------')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = RandomForestRegressor(n_estimators=1000, max_depth=3)
        
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        test_queries = list(set(groups[test_index]))

        for j, query in enumerate(test_queries):
            indices = [k for k, x in enumerate(test_index) if x in np.where(groups == query)[0]]
            if compute_NDCG(y_test[indices], predictions[indices], n=5):
                NDCG_20.append(compute_NDCG(y_test[indices], predictions[indices], n=20))

feature_data = pd.read_csv('data/features.csv')

semantic_features = ['resim', 'remax', 'reavg', 'resum', 'esim', 'emax', 'eavg', 'esum', 
                        'csim', 'cmax', 'cavg', 'csum', 'sim', 'max', 'avg', 'sum']

bag_of_entities = ['esim', 'emax', 'eavg', 'esum']
bag_of_categories = ['csim', 'cmax', 'cavg', 'csum']
word_embeddings = ['sim', 'max', 'avg', 'sum']
graph_embeddings = ['resim', 'remax', 'reavg', 'resum']

X = feature_data.drop(columns=['query_id', 'query','table_id','rel'])
X_baseline = X.drop(columns=semantic_features)
X_bag_of_entities = X.drop(columns=word_embeddings + graph_embeddings + bag_of_categories)
X_bag_of_categories = X.drop(columns=word_embeddings + graph_embeddings + bag_of_entities)
X_word_embeddings = X.drop(columns=bag_of_categories + graph_embeddings + bag_of_entities)
X_graph_embeddings = X.drop(columns=word_embeddings + bag_of_categories + bag_of_entities)

features = X.columns
print(features)
X = X.values
y = feature_data['rel'].values
groups = feature_data['query_id'].values

splits = 5
crossvalidation = GroupKFold(n_splits=splits)

NDCG_5 = []
NDCG_10 = []
NDCG_15 = []
NDCG_20 = []

ERR_5 = []
ERR_10 = []
ERR_15 = []
ERR_20 = []

for i, (train_index, test_index) in enumerate(crossvalidation.split(X, y, groups)):
    print(f'------------ RUNNING MODEL FOR SPLIT {i + 1} / {crossvalidation.get_n_splits()} ------------')
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = RandomForestRegressor(n_estimators=1000, max_depth=3)
    
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    test_queries = list(set(groups[test_index]))

    for j, query in enumerate(test_queries):
        indices = [k for k, x in enumerate(test_index) if x in np.where(groups == query)[0]]
        if compute_NDCG(y_test[indices], predictions[indices], n=5):
            NDCG_5.append(compute_NDCG(y_test[indices], predictions[indices], n=5))
            NDCG_10.append(compute_NDCG(y_test[indices], predictions[indices], n=10))
            NDCG_15.append(compute_NDCG(y_test[indices], predictions[indices], n=15))
            NDCG_20.append(compute_NDCG(y_test[indices], predictions[indices], n=20))

        ERR_5.append(compute_ERR(y_test[indices], predictions[indices], n=5))
        ERR_10.append(compute_ERR(y_test[indices], predictions[indices], n=10))
        ERR_15.append(compute_ERR(y_test[indices], predictions[indices], n=15))            
        ERR_20.append(compute_ERR(y_test[indices], predictions[indices], n=20))

print(NDCG_5)
print(ERR_5)
print(f' ---> Final Scores:')
print(f'NDCG@5 mean: {np.array(NDCG_5).mean()}')
print(f'NDCG@5 std: {np.array(NDCG_5).std()}')
print(f'NDCG@10 mean: {np.array(NDCG_10).mean()}')
print(f'NDCG@10 std: {np.array(NDCG_10).std()}')
print(f'NDCG@15 mean: {np.array(NDCG_15).mean()}')
print(f'NDCG@15 std: {np.array(NDCG_15).std()}')
print(f'NDCG@20 mean: {np.array(NDCG_20).mean()}')
print(f'NDCG@20 std: {np.array(NDCG_20).std()}')

print(f'ERR@5 mean: {np.array(ERR_5).mean()}')
print(f'ERR@5 std: {np.array(ERR_5).std()}')
print(f'ERR@10 mean: {np.array(ERR_10).mean()}')
print(f'ERR@10 std: {np.array(ERR_10).std()}')
print(f'ERR@15 mean: {np.array(ERR_15).mean()}')
print(f'ERR@15 std: {np.array(ERR_15).std()}')
print(f'ERR@20 mean: {np.array(ERR_20).mean()}')
print(f'ERR@20 std: {np.array(ERR_20).std()}')

features = ['Query length', 'IDF field a', 'IDF field b', 'IDF field c', 'IDF field d', 'IDF field e', 'IDF all', '#rows', '#cols',
       '#nulls', 'Inlinks', 'Outlinks', 'Table imp.', 'Page fraction', 'PMI', 'Pageviews',
       'Left col hits', '2nd col hits', 'Body hits', 'Q in title', 'Q in caption',
       'yRank', 'MLM score', 'Word embed. early', 'Word embed. avg', 'Word embed. max', 'Word embed. sum', 
       'Graph embed. early', 'Graph embed. avg',
       'Graph embed. max', 'Graph embed. sum', 'BoC early', 'BoC avg', 'BoC max', 'BoC sum', 'BoE early', 'BoE avg',
       'BoE max', 'BoE sum']
semantic_features = ['Word embed. early', 'Word embed. avg', 'Word embed. max', 'Word embed. sum', 
       'Graph embed. early', 'Graph embed. avg',
       'Graph embed. max', 'Graph embed. sum', 'BoC early', 'BoC avg', 'BoC max', 'BoC sum', 'BoE early', 'BoE avg',
       'BoE max', 'BoE sum']

feature_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)


colors = ['red' if c in semantic_features else 'blue' for c in feature_imp.index]
plt.bar(x=range(len(feature_imp.values)), height=feature_imp.values, tick_label=feature_imp.index, color=colors)
plt.xticks(rotation=90)
# plt.title("Feature importance for the full model")
plt.savefig('feature_imp.pdf', bbox_inches = "tight")