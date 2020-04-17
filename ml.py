import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold

from evaluation_metrics import compute_NDCG, compute_ERR

def get_ndcg20_for_X_Y(X, y, groups, crossvalidation):
    NDCG_20 = []
    np.random.seed(1)
    for i, (train_index, test_index) in enumerate(crossvalidation.split(X, y, groups)):
        # print(f'------------ RUNNING MODEL FOR SPLIT {i + 1} / {crossvalidation.get_n_splits()} ------------')
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

    return np.array(NDCG_20)

def plot_diff(baseline_list, other_list, name):
    plt.hist(other_list - baseline_list, bins=[-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7])
    plt.xlabel('#queries')
    plt.ylabel('difference in NDCG@20')
    plt.savefig(name, bbox_inches = "tight")
    plt.close()

feature_data = pd.read_csv('data/features.csv')

semantic_features = ['resim', 'remax', 'reavg', 'resum', 'esim', 'emax', 'eavg', 'esum', 
                        'csim', 'cmax', 'cavg', 'csum', 'sim', 'max', 'avg', 'sum']

bag_of_entities = ['esim', 'emax', 'eavg', 'esum']
bag_of_categories = ['csim', 'cmax', 'cavg', 'csum']
word_embeddings = ['sim', 'max', 'avg', 'sum']
graph_embeddings = ['resim', 'remax', 'reavg', 'resum']

early = ['esim', 'csim', 'sim','resim']
lsum = ['esum', 'csum', 'sum','resum']
lmax = ['emax', 'cmax', 'max','remax']
lavg= ['eavg', 'cavg', 'avg','reavg']

X = feature_data.drop(columns=['query_id', 'query','table_id','rel'])
X_baseline = X.drop(columns=semantic_features).values
X_bag_of_entities = X.drop(columns=word_embeddings + graph_embeddings + bag_of_categories).values
X_bag_of_categories = X.drop(columns=word_embeddings + graph_embeddings + bag_of_entities).values
X_word_embeddings = X.drop(columns=bag_of_categories + graph_embeddings + bag_of_entities).values
X_graph_embeddings = X.drop(columns=word_embeddings + bag_of_categories + bag_of_entities).values

X_early = X.drop(columns=lsum + lmax + lavg).values
X_lsum = X.drop(columns=lmax + lavg + early).values
X_lmax = X.drop(columns=lsum + early + lavg).values
X_lavg = X.drop(columns=lmax + lsum + early).values


features = X.columns
# X = X.values
y = feature_data['rel'].values
groups = feature_data['query_id'].values

splits = 5
np.random.seed(1)
crossvalidation = GroupKFold(n_splits=splits)

semantic_features = ['resim', 'remax', 'reavg', 'resum', 'esim', 'emax', 'eavg', 'esum', 
                        'csim', 'cmax', 'cavg', 'csum', 'sim', 'max', 'avg', 'sum']

# for feature in semantic_features:
#     temp = semantic_features.copy()
#     temp.remove(feature)
#     temp_X = X.drop(columns=temp).values
#     res = get_ndcg20_for_X_Y(temp_X, y, groups, crossvalidation).mean()
#     print("For " + feature)
#     print(res)

# print("For all BoE")
# print(get_ndcg20_for_X_Y(X_bag_of_entities, y, groups, crossvalidation).mean())
# print("For all BoC")
# print(get_ndcg20_for_X_Y(X_bag_of_categories, y, groups, crossvalidation).mean())
# print("For all WE")
# print(get_ndcg20_for_X_Y(X_word_embeddings, y, groups, crossvalidation).mean())

# print("For all GE")
# print(get_ndcg20_for_X_Y(X_graph_embeddings, y, groups, crossvalidation).mean())

# print("For all early")
# print(get_ndcg20_for_X_Y(X_early, y, groups, crossvalidation).mean())
# print("For all sum")
# print(get_ndcg20_for_X_Y(X_lsum, y, groups, crossvalidation).mean())
# print("For all max")
# print(get_ndcg20_for_X_Y(X_lmax, y, groups, crossvalidation).mean())
# print("For all avg")
# print(get_ndcg20_for_X_Y(X_lavg, y, groups, crossvalidation).mean())

# ndcg_list_baseline = get_ndcg20_for_X_Y(X_baseline, y, groups, crossvalidation)
# ndcg_list_bag_of_entities = get_ndcg20_for_X_Y(X_bag_of_entities, y, groups, crossvalidation)
# ndcg_list_bag_of_categories = get_ndcg20_for_X_Y(X_bag_of_categories, y, groups, crossvalidation)
# ndcg_list_word_embeddings = get_ndcg20_for_X_Y(X_word_embeddings, y, groups, crossvalidation)
# ndcg_list_graph_embeddings = get_ndcg20_for_X_Y(X_graph_embeddings, y, groups, crossvalidation)

# plot_diff(ndcg_list_baseline, ndcg_list_bag_of_entities, 'BoE.pdf')
# plot_diff(ndcg_list_baseline, ndcg_list_bag_of_categories, 'BoC.pdf')
# plot_diff(ndcg_list_baseline, ndcg_list_word_embeddings, 'WE.pdf')
# plot_diff(ndcg_list_baseline, ndcg_list_graph_embeddings, 'GE.pdf')

# NDCG_5 = []
# NDCG_10 = []
# NDCG_15 = []
# NDCG_20 = []

# ERR_5 = []
# ERR_10 = []
# ERR_15 = []
# ERR_20 = []

# for i, (train_index, test_index) in enumerate(crossvalidation.split(X, y, groups)):
#     print(f'------------ RUNNING MODEL FOR SPLIT {i + 1} / {crossvalidation.get_n_splits()} ------------')
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
    
#     model = RandomForestRegressor(n_estimators=1000, max_depth=3)
    
#     model.fit(X_train, y_train)

#     predictions = model.predict(X_test)

#     test_queries = list(set(groups[test_index]))

#     for j, query in enumerate(test_queries):
#         indices = [k for k, x in enumerate(test_index) if x in np.where(groups == query)[0]]
#         if compute_NDCG(y_test[indices], predictions[indices], n=5):
#             NDCG_5.append(compute_NDCG(y_test[indices], predictions[indices], n=5))
#             NDCG_10.append(compute_NDCG(y_test[indices], predictions[indices], n=10))
#             NDCG_15.append(compute_NDCG(y_test[indices], predictions[indices], n=15))
#             NDCG_20.append(compute_NDCG(y_test[indices], predictions[indices], n=20))

#         ERR_5.append(compute_ERR(y_test[indices], predictions[indices], n=5))
#         ERR_10.append(compute_ERR(y_test[indices], predictions[indices], n=10))
#         ERR_15.append(compute_ERR(y_test[indices], predictions[indices], n=15))            
#         ERR_20.append(compute_ERR(y_test[indices], predictions[indices], n=20))

# print(NDCG_5)
# print(ERR_5)
# print(f' ---> Final Scores:')
# print(f'NDCG@5 mean: {np.array(NDCG_5).mean()}')
# print(f'NDCG@5 std: {np.array(NDCG_5).std()}')
# print(f'NDCG@10 mean: {np.array(NDCG_10).mean()}')
# print(f'NDCG@10 std: {np.array(NDCG_10).std()}')
# print(f'NDCG@15 mean: {np.array(NDCG_15).mean()}')
# print(f'NDCG@15 std: {np.array(NDCG_15).std()}')
# print(f'NDCG@20 mean: {np.array(NDCG_20).mean()}')
# print(f'NDCG@20 std: {np.array(NDCG_20).std()}')

# print(f'ERR@5 mean: {np.array(ERR_5).mean()}')
# print(f'ERR@5 std: {np.array(ERR_5).std()}')
# print(f'ERR@10 mean: {np.array(ERR_10).mean()}')
# print(f'ERR@10 std: {np.array(ERR_10).std()}')
# print(f'ERR@15 mean: {np.array(ERR_15).mean()}')
# print(f'ERR@15 std: {np.array(ERR_15).std()}')
# print(f'ERR@20 mean: {np.array(ERR_20).mean()}')
# print(f'ERR@20 std: {np.array(ERR_20).std()}')

# features = ['Query length', 'IDF field a', 'IDF field b', 'IDF field c', 'IDF field d', 'IDF field e', 'IDF all', '#rows', '#cols',
#        '#nulls', 'Inlinks', 'Outlinks', 'Table imp.', 'Page fraction', 'PMI', 'Pageviews',
#        'Left col hits', '2nd col hits', 'Body hits', 'Q in title', 'Q in caption',
#        'yRank', 'MLM score', 'Word embed. early', 'Word embed. avg', 'Word embed. max', 'Word embed. sum', 
#        'Graph embed. early', 'Graph embed. avg',
#        'Graph embed. max', 'Graph embed. sum', 'BoC early', 'BoC avg', 'BoC max', 'BoC sum', 'BoE early', 'BoE avg',
#        'BoE max', 'BoE sum']
# semantic_features = ['Word embed. early', 'Word embed. avg', 'Word embed. max', 'Word embed. sum', 
#        'Graph embed. early', 'Graph embed. avg',
#        'Graph embed. max', 'Graph embed. sum', 'BoC early', 'BoC avg', 'BoC max', 'BoC sum', 'BoE early', 'BoE avg',
#        'BoE max', 'BoE sum']

# feature_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)


# colors = ['red' if c in semantic_features else 'blue' for c in feature_imp.index]
# plt.bar(x=range(len(feature_imp.values)), height=feature_imp.values, tick_label=feature_imp.index, color=colors)
# plt.xticks(rotation=90)
# # plt.title("Feature importance for the full model")
# plt.savefig('feature_imp.pdf', bbox_inches = "tight")