from utils import get_entity_to_information_dict, write_dictionary_to_file
from tqdm import tqdm

print('----- START READING INFORMATION FILE -----')
dict_information = get_entity_to_information_dict('../dictionaries/entities_to_information.csv')
print('----- FINISHED READING INFORMATION FILE -----')

category_index_list = []
entity_index_list = []


print('---- Reading all to create indices')
for k, v in tqdm(dict_information.items()):
    category_index_list += v['categories']
    entity_index_list += v['outlinks'] + v['inlinks'] + [k]

category_index_list = [x.lower() for x in category_index_list]
entity_index_list = [x.lower() for x in entity_index_list]
category_index_list = list(set(category_index_list))
entity_index_list = list(set(entity_index_list))

category_index_dict = dict((c, i) for i, c in enumerate(category_index_list))
entity_index_dict = dict((c, i) for i, c in enumerate(entity_index_list))

category_index_list = 0
entity_index_list = 0

entities_dictionaries = {}
categories_dictionaries = {}


print('---- Finding all indices')
for k, v in tqdm(dict_information.items()):
    entity_indices = []
    category_indices = []

    for category in v['categories']:
        category_indices.append(category_index_dict[category.lower()])
    for entity in set(v['outlinks'] + v['inlinks'] + [k]):
        entity_indices.append(entity_index_dict[entity.lower()])
    
    categories_dictionaries[k.lower()] = {}
    for index in category_indices:
        categories_dictionaries[k.lower()][index] = 1

    entities_dictionaries[k.lower()] = {}
    for index in entity_indices:
        entities_dictionaries[k.lower()][index] = 1


write_dictionary_to_file(entities_dictionaries, '../dictionaries/link_indices.json')
write_dictionary_to_file(categories_dictionaries, '../dictionaries/category_indices.json')