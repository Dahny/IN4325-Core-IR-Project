from nltk import word_tokenize
import re
import string
import json

dict_entity_tables = {}
dict_table_entities = {}
dict_word_entity = {}
counters = {'table_entities': 0, 'entity_tables': 0, 'word_entity': 0}


def create_dictionaries_from_wiki_tables(wiki_tables, path_name):

    dict_headers = {}
    dict_page_titles = {}
    dict_captions = {}
    dict_section_titles = {}
    dict_data = {}

    for wiki_table in wiki_tables:
        preprocessed_page_title = preprocess_string(wiki_table['pgTitle'])
        extract_entity_and_add_to_dict(wiki_table['pgTitle'], wiki_table['table_id'])
        list(map(lambda x: add_to_dict(dict_page_titles, x, wiki_table['table_id']), preprocessed_page_title))

        preprocessed_section_title = preprocess_string(wiki_table['secondTitle'])
        extract_entity_and_add_to_dict(wiki_table['secondTitle'], wiki_table['table_id'])
        list(map(lambda x: add_to_dict(dict_section_titles, x, wiki_table['table_id']), preprocessed_section_title))

        preprocessed_caption = preprocess_string(wiki_table['caption'])
        extract_entity_and_add_to_dict(wiki_table['caption'], wiki_table['table_id'])
        list(map(lambda x: add_to_dict(dict_captions, str(x), str(wiki_table['table_id'])), preprocessed_caption))

        preprocessed_headers = list(map(lambda x: ' '.join(preprocess_string(x)), wiki_table['title']))
        list(map(lambda x: extract_entity_and_add_to_dict(x, wiki_table['table_id']), wiki_table['title']))
        list(map(lambda x: add_to_dict(dict_headers, str(x), str(wiki_table['table_id'])), preprocessed_headers))

        preprocessed_data = list(map(lambda x: list(map(lambda y: preprocess_string(y), x)), wiki_table['data']))
        list(map(lambda x: list(map(lambda y: extract_entity_and_add_to_dict(y, wiki_table['table_id']), x)),
                 wiki_table['data']))
        list(map(lambda x: list(map(lambda y: list(map(lambda z: add_to_dict(dict_data, z, wiki_table['table_id']),
                                                       y)), x)), preprocessed_data))

    print("counters ", counters)

    write_dictionary_to_file(dict_headers, path_name + '/dictionaries/dictionary_headers.json')
    write_dictionary_to_file(dict_page_titles, path_name + '/dictionaries/dictionary_page_titles.json')
    write_dictionary_to_file(dict_section_titles, path_name + '/dictionaries/dictionary_section_titles.json')
    write_dictionary_to_file(dict_captions, path_name + '/dictionaries/dictionary_captions.json')
    write_dictionary_to_file(dict_data, path_name + '/dictionaries/dictionary_data.json')
    write_dictionary_to_file(dict_entity_tables, path_name + '/dictionaries/dictionary_entity_tables.json')
    write_dictionary_to_file(dict_table_entities, path_name + '/dictionaries/dictionary_table_entities.json')
    write_dictionary_to_file(dict_word_entity, path_name + '/dictionaries/dictionary_word_entity.json')


def create_dictionaries_from_queries(queries, path_name):
    dict_queries = {}
    for query in queries:
        preprocessed_query = preprocess_string(query)
        list(map(lambda x: add_to_dict(dict_queries, x, query['query_id']), preprocessed_query))

    write_dictionary_to_file(dict_queries, path_name + '/dictionaries/dictionary_queries.json')


def write_dictionary_to_file(dictionary, file_name):
    with open(file_name, 'w') as f:
        json.dump(dictionary, f)
        f.close()


def preprocess_string(s):
    pre_s = re.sub('[\(\[].*?[\|\)]', '', s.lower())
    pre_s = pre_s.translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(pre_s)


def extract_entity_and_add_to_dict(s, table_id):

    entity = re.search(r'\[.*?\|', s)
    corresponding_word = re.search(r'\|.*?\]', s)

    if entity is not None:
        str_entity = entity.group(0).replace('[', '').replace('|', '')
        str_corresponding_word = corresponding_word.group(0).replace(']', '').replace('|', '').lower()

        if table_id not in dict_table_entities:
            dict_table_entities[table_id] = [str(counters['table_entities']), str_entity]
            counters['table_entities'] += 1
        else:
            if str_entity not in dict_table_entities[table_id]:
                dict_table_entities[table_id].append(str_entity)

        if str_entity not in dict_entity_tables:
            dict_entity_tables[str_entity] = [str(counters['entity_tables']), table_id]
            counters['entity_tables'] += 1
        else:
            if table_id not in dict_entity_tables[str_entity]:
                dict_entity_tables[str_entity].append(table_id)

        if str_corresponding_word not in dict_word_entity:
            dict_word_entity[str_corresponding_word] = [str(counters['word_entity']), str_entity]
            counters['word_entity'] += 1
        else:
            if str_entity not in dict_word_entity[str_corresponding_word]:
                dict_word_entity[str_corresponding_word].append(str_entity)


def add_to_dict(dictionary, word, table_id):

    if word not in dictionary:
        dictionary[word] = [table_id]
    else:
        if table_id not in dictionary[word]:
            dictionary[word].append(table_id)



table = [{
        "table_id": "table-0001-249",
		"title": ["Athlete", "Nation", "Total", "Gold", "Silver", "Bronze", "Events"],
		"numCols": 7,
		"numericColumns": [2, 3, 4, 5],
		"pgTitle": "Auburn Tigers swimming and diving",
		"numDataRows": 6,
		"secondTitle": "Summer Olympic Games Beijing 2008",
		"numHeaderRows": 1,
		"caption": "Summer Olympic Games Beijing 2008",
		"data": [
			["[Fr\u00e9d\u00e9rick_Bousquet|Fr\u00e9d\u00e9rick Bousquet]", "[France|FRA]", "1", "0", "1", "0", "Silver Medal in 400m Freestyle relay"],
			["[C\u00e9sar_Cielo|C\u00e9sar Cielo Filho]", "[Brazil|BRA]", "2", "1", "0", "1", "Gold Medal in 50m Freestyle and Bronze Medal for 100m Freestyle"],
			["[Kirsty_Coventry|Kirsty Coventry]", "[Zimbabwe|ZIM]", "4", "1", "3", "0", "Gold Medal in 200m backstroke, Silver Medal in 100m basckstroke, 200m Medley and 400m Medley."],
			["[Mark_Gangloff|Mark Gangloff]", "[United_States|USA]", "1", "1", "0", "0", "Gold Medal in 400m Medley Relay"],
			["[Margaret_Hoelzer|Margaret Hoelzer]", "[United_States|USA]", "3", "0", "2", "1", "Silver Medal in 200m backstroke, 400m Medley, Bronze Medal in 100m backstroke"],
			["[Matt_Targett|Matt Targett]", "[Australia|AUS]", "2", "0", "1", "1", "Silver Medal in 400m Medley Relay, Bronze Medal in 400m freestyle relay"]
		]
	},{
        "table_id": "table-0001-400",
		"title": ["Dose (\u00b5g/kg/day)", "[Environmental_Working_Group|Environmental Working Group]", "Study Year"],
		"numCols": 3,
		"numericColumns": [0, 2],
		"pgTitle": "Bisphenol A",
		"numDataRows": 12,
		"secondTitle": "Low-dose exposure in animals",
		"numHeaderRows": 1,
		"caption": "Low-dose exposure in animals",
		"data": [
			["0.025", "\"Permanent changes to genital tract\"", "2005"],
			["0.025", "\"Changes in breast tissue that predispose cells to hormones and carcinogens\"", "2005"],
			["1", "long-term adverse reproductive and carcinogenic effects", "2009"],
			["2", "\"increased prostate weight 30%\"", "1997"],
			["2", "[Anogenital_distance|anogenital distance]", "[Digital_object_identifier|doi]"],
			["2.4", "\"Decline in testicular testosterone\"", "2004"],
			["2.5", "\"Breast cells predisposed to cancer\"", "2007"],
			["10", "\"Prostate cells more sensitive to hormones and cancer\"", "2006"],
			["10", "\"Decreased maternal behaviors\"", "2002"],
			["30", "\"Reversed the normal sex differences in brain structure and behavior\"", "2003"],
			["50", "[Primate|non-human primates]", "2008"],
			["50", "Disrupts ovarian development", "2009"]
		]
	}
]


if __name__ == '__main__':
    path_name = '/Users/lisameijer/Universiteit/master_1_2019_2020/InformationRetrieval/project_IR/IN4325-Core-IR-Project'
    create_dictionaries_from_wiki_tables(table, path_name)