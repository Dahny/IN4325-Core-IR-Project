import pandas as pd
import os
import json
import baselineFeatures


def read_items(input_file: str):
    items = {}
    with open(input_file, 'rb') as file:
        for line in file:
            item = json.loads(line.decode('utf-8'))
            items[int(item["id"])] = item
    return items


def write_items(items, output_file):
    df = pd.DataFrame.from_dict(items, orient="index")
    df.to_csv(output_file, sep=',', index=False)


def feature_extraction(input_file: str, output_file: str):

    items_in = read_items(input_file)
    items_out = {}

    if os.path.isfile(output_file):
        print("Target output file exists, reading existing data.")
        items_out = pd.read_csv(output_file).set_index('id').to_dict(orient="index")

    for id in items_in:
        if id not in items_out:
            items_out[id] = {"id": id}
        else:
            items_out[id]["id"] = id

        baselineFeatures.compute_baseline_features(items_in[id], items_out[id])

    print("Computing all features completed")

    write_items(items_out, output_file)