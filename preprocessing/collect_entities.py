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