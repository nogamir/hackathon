import sys

sys.path.append("../")
import numpy as np
import pandas as pd
from collections import defaultdict

# consts
NA = "Na"


def load_data(path_to_file):
    """
    this function loads a data set and preforms all the necessary
    preprocessing inorder to get a valid design matrix
    :param path_to_file: path to file
    :return: pandas DataFrame
    """
    df = pd.read_csv(path_to_file)
    # df.insert(0, 0, 1)  # adding intersect
    # df = df.loc[df["revenue"] > 0]
    df["belongs_to_collection_name"] = df.apply(lambda row: handle_belongs_to_collection(row.belongs_to_collection, "name"), axis=1)
    # split_feature_to_cols(df, "genres", "name")
    split_feature_to_cols(df, "production_companies", "name")
    return df


def handle_belongs_to_collection(s, field):
    if type(s) == str:
        d = eval(s)
        return d[field]
    return NA


def split_feature_to_cols(df, col_name, field_name):
    d = defaultdict(list)
    df[col_name] = df.apply(lambda row: from_list_to_cols(row, d, row[col_name], field_name), axis=1)
    data = np.array(list(d.items()))
    a = np.apply_along_axis(lambda row: len(row[1]), 1, data).reshape(-1, 1)
    data = np.concatenate((data, a),axis=1)
    print(data)



def from_list_to_cols(row, main_d, lst, field_name):
    if type(lst) == str:
        for d in eval(lst):
            main_d[d[field_name]].append(row["id"])


if __name__ == '__main__':
    path = "movies_dataset.csv"
    X = load_data(path)
    print(X.head())
    print(X["genres"])
