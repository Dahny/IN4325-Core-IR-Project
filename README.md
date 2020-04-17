# IN4325-Core-IR-Project
This is the repository for the Core IR Project regarding paper: https://arxiv.org/pdf/1802.06159.pdf 

# Code Structure:
## Data
This is where we keep all of the raw data files

## Dictionaries
This is where we keep all of the dictionaries files for the semantic features

## Preprocessing
This is where we apply all the preprocessing of the data before we use it for the models

## Main
Baseline_features.py
This is where the baseline features are computer
Evaluation_metrics.py
This is where we compute the NDCG and ERR scores
extraction.py
This is where we extract the features from the data
ml.py
This is where the RandomRegressor is used for our LTR implementations
semantic-features.py
This is where the semantic features are computed
table-extractor.py
This was used to extract the relevant tables for the qrels file
unsupervised_ranking.py
This is where we setup the unsupervised models with elasticsearch and compute the rankings for the single and multi-field approaches

# Setup Unsupervised ranking
In order to run the unsupervised ranking methods a running Elasticsearch server is required. In our case we downloaded a docker image and ran a container using the docker-compose.yml file in the main directory. After setting up the docker container run the pipeline method in unsupervised_ranking.py and wait untill it is done.

