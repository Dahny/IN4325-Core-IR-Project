from smart_open import open
import numpy as np
import json
from tqdm import tqdm
from utils import write_dictionary_to_file

base_url = "http://data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/"
url = base_url + "2016-04/GlobalVectors/1_uniform/DBpediaVecotrs200_20Shuffle.txt"

with open('../dictionaries/table_to_entities.json', 'r') as f:
    table_entities = json.loads(f.read())

with open('../dictionaries/query_to_entities.json', 'r') as f:
    query_entities = json.loads(f.read())

entities_list = []
for k, v in table_entities.items():
    entities_list += v

for k, v in query_entities.items():
    entities_list += v

entities_list = list(map(lambda x: x.lower(), set(entities_list)))

vecs = {}

counter = 0

print(len(entities_list))

warnings = []

for line in open(url):
    try:
        preprocessed = line.split('http://dbpedia.org/resource/')[1].split('>')
        entity_name = preprocessed[0].strip()
        vector = preprocessed[1].strip()
        if entity_name.lower() in entities_list:
            print(f'---- Found entity {entity_name}')
            vecs[entity_name] = np.fromstring(vector, sep=" ")
    except:
        print('Warning, no entity found.')
        warnings.append(line)
        pass
    counter += 1
    if counter % 500 == 0:
        print(f'---- Processed {counter} entities')

print('WARNINGS:')
print(warnings)
print('Done!')

write_dictionary_to_file(vecs, '../dictionaries/rdf2vec.json')