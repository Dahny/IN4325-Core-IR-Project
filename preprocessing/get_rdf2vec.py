from smart_open import open
import numpy as np
import json
from tqdm import tqdm
from utils import write_dictionary_to_file

base_url = "http://data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/"
url = base_url + "2016-04/GlobalVectors/1_uniform/DBpediaVecotrs200_20Shuffle.txt"

# replace this with actual entities we are looking for
entities_list = ['European_Union', 'Council_of_the_European_Union', 'Underground_Ernie']
entities_list = list(map(lambda x: x.lower(), entities_list))

vecs = {}

for line in tqdm(open(url)):
    preprocessed = line.split('http://dbpedia.org/resource/')[1].split('>')
    entity_name = preprocessed[0].strip()
    vector = preprocessed[1].strip()
    if entity_name.lower() in entities_list:
        vecs[entity_name] = np.fromstring(vector, sep=" ")


write_dictionary_to_file(vecs, '../dictionaries/rdf2vec.json')
with open('rdf2vec.json', 'w') as json_file:
    json.dump(vecs, json_file)