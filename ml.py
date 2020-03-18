import pandas as pd

feature_data = pd.read_csv('features.csv')

X = feature_data.drop(columns=['query_id','query','table_id','rel']).values
Y = feature_data['rel'].values