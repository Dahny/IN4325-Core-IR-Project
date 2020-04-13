import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold


feature_data = pd.read_csv('data/features.csv')

X = feature_data.drop(columns=['query_id', 'query','table_id','rel'])
features = X.columns
X = X.values
y = feature_data['rel'].values
groups = feature_data['query_id'].values

crossvalidation = GroupKFold(n_splits=5)

NDCG_10_results = np.zeros(5)

for i, (train_index, test_index) in enumerate(crossvalidation.split(X, y, groups)):
  print(f'------------ RUNNING MODEL FOR SPLIT {i + 1} / {crossvalidation.get_n_splits()} ------------')
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  
  model = RandomForestRegressor(n_estimators=1000, max_depth=3)
  
  model.fit(X_train, y_train)

  predictions = model.predict(X_test)

  test_queries = list(set(groups[test_index]))

  scores = np.zeros(len(test_queries))

  for j, query in enumerate(test_queries):
    indices = [k for k, x in enumerate(test_index) if x in np.where(groups == query)[0]]
    scores[j] = ndcg_score([y_test[indices]], [predictions[indices]], k=10)

  NDCG_10_results[i] = scores.mean()
  print(f' ---> NDCG@10Score was {scores.mean()}')

print(f' ---> Final NDCG@10Score: {NDCG_10_results.mean()}')

feature_imp = pd.Series(model.feature_importances_,index=features).sort_values(ascending=False)

semantic_features = ['resim', 'remax', 'reavg', 'resum', 'esim', 'emax', 'eavg', 'esum', 
                        'csim', 'cmax', 'cavg', 'csum', 'sim', 'max', 'avg', 'sum']

colors = ['red' if c in semantic_features else 'blue' for c in feature_imp.index]

plt.bar(x=range(len(feature_imp.values)), height=feature_imp.values, tick_label=feature_imp.index, color=colors)
plt.show()