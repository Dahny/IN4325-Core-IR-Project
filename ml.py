import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold

from evaluation_metrics import compute_NDCG, compute_ERR

feature_data = pd.read_csv('data/features.csv')

X = feature_data.drop(columns=['query_id', 'query','table_id','rel'])
features = X.columns
X = X.values
y = feature_data['rel'].values
groups = feature_data['query_id'].values

splits = 5
crossvalidation = GroupKFold(n_splits=splits)

NDCG_5 = np.zeros(len(set(groups)))
NDCG_10 = np.zeros(len(set(groups)))
NDCG_15 = np.zeros(len(set(groups)))
NDCG_20 = np.zeros(len(set(groups)))

ERR_5 = np.zeros(len(set(groups)))
ERR_10 = np.zeros(len(set(groups)))
ERR_15 = np.zeros(len(set(groups)))
ERR_20 = np.zeros(len(set(groups)))
score_index = 0

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
        NDCG_5[score_index] = compute_NDCG(y_test[indices], predictions[indices], n=5)
        ERR_5[score_index] = compute_ERR(y_test[indices], predictions[indices], n=5)        
        NDCG_10[score_index] = compute_NDCG(y_test[indices], predictions[indices], n=10)
        ERR_10[score_index] = compute_ERR(y_test[indices], predictions[indices], n=10)        
        NDCG_15[score_index] = compute_NDCG(y_test[indices], predictions[indices], n=15)
        ERR_15[score_index] = compute_ERR(y_test[indices], predictions[indices], n=15)        
        NDCG_20[score_index] = compute_NDCG(y_test[indices], predictions[indices], n=20)
        ERR_20[score_index] = compute_ERR(y_test[indices], predictions[indices], n=20)
        score_index += 1

print(NDCG_5)
print(ERR_5)
print(f' ---> Final Scores:')
print(f'NDCG@5 mean: {NDCG_5.mean()}')
print(f'NDCG@5 std: {NDCG_5.std()}')
print(f'NDCG@10 mean: {NDCG_10.mean()}')
print(f'NDCG@10 std: {NDCG_10.std()}')
print(f'NDCG@15 mean: {NDCG_15.mean()}')
print(f'NDCG@15 std: {NDCG_15.std()}')
print(f'NDCG@20 mean: {NDCG_20.mean()}')
print(f'NDCG@20 std: {NDCG_20.std()}')

print(f'ERR@5 mean: {ERR_5.mean()}')
print(f'ERR@5 std: {ERR_5.std()}')
print(f'ERR@10 mean: {ERR_10.mean()}')
print(f'ERR@10 std: {ERR_10.std()}')
print(f'ERR@15 mean: {ERR_15.mean()}')
print(f'ERR@15 std: {ERR_15.std()}')
print(f'ERR@20 mean: {ERR_20.mean()}')
print(f'ERR@20 std: {ERR_20.std()}')

feature_imp = pd.Series(model.feature_importances_,index=features).sort_values(ascending=False)

semantic_features = ['resim', 'remax', 'reavg', 'resum', 'esim', 'emax', 'eavg', 'esum', 
                        'csim', 'cmax', 'cavg', 'csum', 'sim', 'max', 'avg', 'sum']

colors = ['red' if c in semantic_features else 'blue' for c in feature_imp.index]

plt.bar(x=range(len(feature_imp.values)), height=feature_imp.values, tick_label=feature_imp.index, color=colors)
plt.show()