import os
import pandas as pd
import json

qrels = pd.read_csv('data/qrels.txt', sep='\t', header=None)
tables = qrels.iloc[:,2]
output = {}
amount = 0

for root, dirs, files in os.walk("tables_redi2_1"):
    for file in files:
        with open(root + '/' + file) as json_file:
            data = json.load(json_file)
            for table1 in data:
                for table2 in tables:
                    if table1 == table2:
                        amount += 1
                        print('found table:',table1, 'total amount of', amount, 'tables now')
                        output[table1] = data[table1]
                        break

with open('data/raw_table_data.json', 'w') as outfile:
    json.dump(output, outfile)
