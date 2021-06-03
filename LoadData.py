import sys

sys.path.append("../")
import numpy as np
import pandas as pd
from collections import defaultdict
import time

# consts
NA = "Na"


def load_data(path):
    df = pd.read_csv(path)
    # df.insert(0, 0, 1)  # adding intersect
    # df = df.loc[df["revenue"] > 0]
    df["belongs_to_collection"] = '[' + df['belongs_to_collection'].astype(str) + ']'
    return df


def get_score_mats(df, features):
    """
    this function loads a data set and preforms all the necessary
    preprocessing inorder to get a valid design matrix
    :param path_to_file: path to file
    :return: pandas DataFrame
    """
    mats = []
    for col_name, field_name, p in features:
        print(col_name)
        d = get_col_dict_count(df, col_name, field_name)
        score_mat = get_score_mat(d, p, df)
        d = defaultdict(list)
        for row in score_mat:
            d[row[0]] = row[1:]

        mats.append(d)
    return mats


def generate_score_cols(df, features, mats, mode):
    for i, f in enumerate(features):
        feature_name, field_name, p = f
        mat = mats[i]
        value_vec = df.apply(lambda row: calc_cell_val(mat, row[feature_name], field_name, mode), axis=1)
        vec_name = "score_value_" + feature_name
        df[vec_name] = value_vec


def calc_cell_val(score_mat, cell, field_name, mode=1):  # 0 for rating, 1 for revenue
    val = 0
    if cell:
        for d in cell:
            if d[field_name] in score_mat:
                val += score_mat[d[field_name]][mode]
    return val


def remove_features_cols(df, features):
    for feature_name, field_name, p in features:
        df.drop(feature_name, axis='columns', inplace=True)
    return df


def get_col_dict_count(df, col_name, field_name):
    d = defaultdict(list)
    df[col_name] = df.apply(lambda row: create_dict_from_col(df, col_name, row, d, row[col_name], field_name), axis=1)
    data = np.array(list(d.items()))
    a = np.apply_along_axis(lambda row: len(row[1]), 1, data).reshape(-1, 1)
    data = np.concatenate((data, a), axis=1)
    return data


def create_dict_from_col(df, col_name, row, main_d, lst, field_name):
    if type(lst) == str:
        l = eval(lst)
        for d in l:
            main_d[d[field_name]].append(row["id"] - 1)
        return l


def get_score_mat(new_features, percentile, df):
    score_mat = get_common(new_features, percentile)
    for feature in score_mat:
        col = np.zeros(len(df))
        np.put(col, feature[1], 1)
        feature[1] = np.corrcoef(col, df['vote_average'])[0, 1]
        feature[2] = np.corrcoef(col, df['revenue'])[0, 1]
    return score_mat


def get_common(new_features, percentile):
    #  find threshold:
    feature_frequency_col = new_features[:, 2]
    threshold = np.percentile(feature_frequency_col, percentile, axis=0)
    #  return common features list:
    common_features = new_features[feature_frequency_col >= threshold]
    return common_features


def create_dummies_from_language():
    global df
    # language
    counts = pd.value_counts(df["original_language"])
    mask = df["original_language"].isin(counts[counts > 29].index)
    dummies = pd.get_dummies(df["original_language"][mask])
    df.drop("original_language", axis='columns', inplace=True)
    df = pd.concat([df, dummies])


def get_data(mode=1):  # 0 for rating, 1 for revenue
    s_time = time.time()
    path = "movies_dataset.csv"
    df = load_data(path)
    features = [("genres", "name", 0), ("production_companies", "name", 99), ("keywords", "name", 90)
        ,("cast", "name", 90), ("crew", "name", 90)]
    mats = get_score_mats(df, features)
    generate_score_cols(df, features, mats,mode)
    remove_features_cols(df, features)
    create_dummies_from_language()
    e_time = time.time()
    print(f"Run-time is:{e_time - s_time}")
    return df
