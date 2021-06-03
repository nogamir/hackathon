import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_num_of_words(str):
    return len(str.split(" "))


def get_year(str):
    return int(str.split("/")[2])


def get_month(str):
    return int(str.split("/")[1])


def create_dict(col):
    pass


def load_data(path):
    df = pd.read_csv(path)
    df = df.loc[df.revenue > 0]  # temp

    df["title_length"] = df.apply(lambda row: get_num_of_words(row.original_title), axis=1)

    df["month"] = df.apply(lambda row: get_month(row.release_date), axis=1)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["year"] = df.apply(lambda row: get_year(row.release_date), axis=1)+(df["month"]-1)/12

    # df = pd.get_dummies(df, columns=['original_language'], drop_first=True)

    # df = df.drop(labels='id', axis='columns')
    # df.date = pd.to_numeric(df.date.str[:8], errors='coerce')
    # df = df.dropna(axis=0, how='any')
    # df.insert(0, 'intercept', 1.0, allow_duplicates=True)
    return df





if __name__ == '__main__':
    df = load_data("movies_dataset.csv")
    print(df.head())
