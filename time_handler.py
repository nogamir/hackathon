import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_num_words(str):
    return len(str.split(" "))


def get_year(str):
    if "/" not in str:
        print(str)
        return 1897
    return int(str.split("/")[2])


def get_month(str):
    if "/" not in str:
        print(str)
        return 10
    return int(str.split("/")[1])


def add_time_features(df):
    df["release_date"] = df["release_date"].fillna("1/1/1990")
    df["month"] = df.apply(lambda row: get_month(row.release_date), axis=1)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["year"] = df.apply(lambda row: get_year(row.release_date), axis=1) + (df["month"] - 1) / 12
    df = df.drop(["release_date"], axis=1)
    return df


def make_sense(df):
    initial_num_rows = df.shape[0]
    print('initial rows num:', initial_num_rows)
    # df = df.loc[df["belongs_to_collection"]  np.nan]
    print("num rows with cond1:", df.shape[0])
    # df = df.loc[df["release date"] >= val]
    # print(f"{df.shape[0]} samples with two conditions")

#
# if __name__ == '_main_':
#     # %%
#     df = pd.read_csv("movies_dataset.csv")
#     # df = add_time_features(df)
#     # %%
#     # handle_corrupted(df)
#     make_sense(df)
#     # print(df.head())