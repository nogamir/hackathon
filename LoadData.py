import sys

sys.path.append("../")
import numpy as np
import pandas as pd

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
    df = df.loc[df["revenue"] > 0]
    df["belongs_to_collection_name"] = df.apply(lambda row: get_feature_from_text_dict(row.belongs_to_collection, "name"), axis=1)
    return df


def get_feature_from_text_dict(s, field):
    if type(s) == str:
        d = eval(s)
        return d[field]
    return NA


if __name__ == '__main__':
    path = "movies_dataset.csv"
    X = load_data(path)
    print(X.head())
