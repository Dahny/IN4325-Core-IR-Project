

def compute_baseline_features(item_in, item_out):
    number_of_rows(item_in, item_out)
    number_of_columns(item_in, item_out)


def number_of_rows(item_in, item_out):
    item_out[f"nRows"] = item_in['numDataRows']


def number_of_columns(item_in, item_out):
    item_out[f"nColumns"] = item_in['numCols']