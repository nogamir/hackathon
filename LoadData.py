import sys

sys.path.append("../")
import numpy as np
import pandas as pd
from collections import defaultdict
import time
import time_handler
# consts
NA = "Na"
FEATURES = [("production_countries", "name", 80), ("genres", "name", 0), ("production_companies", "name", 80),
            ("keywords", "name", 80), ("cast", "name", 80), ("crew", "name", 80)]
mask = []

# def set_feature(p):
#     global FEATURES
#     FEATURES = [("production_countries", "name", p), ("genres", "name", 10), ("production_companies", "name", p),
#                 ("keywords", "name", p), ("cast", "name", p), ("crew", "name", p)]



def load_data(path):
    df = pd.read_csv(path)
    # df.insert(0, 0, 1)  # adding intercept
    # df["belongs_to_collection"] = '[' + df['belongs_to_collection'].astype(str) + ']'
    df.dropna(subset=["release_date"], inplace=True)
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
        value_vec = df.apply(lambda row: calc_cell_val(mat, eval(row[feature_name]), field_name, mode) if type(row[
            feature_name]) == str else calc_cell_val(mat, row[feature_name], field_name, mode), axis=1)
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
            main_d[d[field_name]].append(row["rowIndex"])
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


def create_dummies_from_language_Train(df):
    # language
    global mask
    counts = pd.value_counts(df["original_language"])
    mask = counts[counts > 29]
    dummies = pd.get_dummies(df["original_language"])
    dummies = dummies[mask.index]
    df = pd.concat((df, dummies), axis=1)
    df.drop("original_language", axis='columns', inplace=True)
    return df


def create_dummies_from_language_Test(df):
    # language
    global mask
    for lang in mask.index:
        vec = np.where(df["original_language"] == lang, 1, 0)
        df[lang] = vec
    df.drop("original_language", axis='columns', inplace=True)
    return df

def get_df_and_mats():
    path = "movies_dataset.csv"
    df = load_data(path)
    df['rowIndex'] = np.arange(df.shape[0])
    mats = get_score_mats(df, FEATURES)
    # mats = []
    df.drop("rowIndex", axis='columns', inplace=True)
    return df, mats


def modify_original_title(df):
    df["original_title"] = df.apply(lambda row: len(row.original_title.split(" ")) if type(row.original_title) is str else 0, axis=1)
    return df


def modify_data(df, mats, mode=1):  # 0 for rating, 1 for revenue
    # for each data
    generate_score_cols(df, FEATURES, mats, mode)
    remove_features_cols(df, FEATURES)
    df = create_dummies_from_language_Train(df)
    df = time_handler.add_time_features(df)
    df = modify_original_title(df)
    vec = np.where(df["belongs_to_collection"] == df["belongs_to_collection"], 1, 0)
    df["belongs_to_collection"] = vec
    df.drop(columns=["id", "homepage", "overview", "title", "tagline", "spoken_languages"], inplace=True)
    return df

def modify_test(df, mats, mode=1):  # 0 for rating, 1 for revenue
    # for each data
    generate_score_cols(df, FEATURES, mats, mode)
    remove_features_cols(df, FEATURES)
    df = create_dummies_from_language_Test(df)
    df = time_handler.add_time_features(df)  # median Year
    df = modify_original_title(df)
    vec = np.where(df["belongs_to_collection"] == df["belongs_to_collection"], 1, 0)
    df["belongs_to_collection"] = vec
    df.drop(columns=["id", "homepage", "overview", "title", "tagline", "spoken_languages"],inplace=True)
    return df

