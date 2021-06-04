import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def load_data(path):
    df = pd.read_csv(path)
    df = df.loc[df.revenue > 0]  # temp

    df["title_length"] = df.apply(lambda row: get_num_of_words(row.original_title), axis=1)
    df["year"] = df.apply(lambda row: get_year(row.release_date), axis=1)
    df["month"] = df.apply(lambda row: get_month(row.release_date), axis=1)
    df = pd.get_dummies(df, columns=['original_language'], drop_first=True)

    # df = df.drop(labels='id', axis='columns')
    # df.date = pd.to_numeric(df.date.str[:8], errors='coerce')
    # df = df.dropna(axis=0, how='any')
    # df.insert(0, 'intercept', 1.0, allow_duplicates=True)
    return df


def get_num_of_words(str):
    return len(str.split(" "))

def get_year(str):
    return (int)(str.split("/")[2])

def get_month(str):
    return (int)(str.split("/")[1])

def sum_column(col):
    pass

def remove_rare_columns(new_features, percentile):
    #  find threshold:
    feature_frequency_col = new_features[:, 2]
    threshold = np.percentile(feature_frequency_col, percentile, axis=0)
    #  return common features list:
    common_features = new_features[feature_frequency_col >= threshold]
    sum = common_features[:, 2].sum()
    common_features[:, 2] = common_features[:, 2] / sum
    return common_features[:, [0, 2]]




if __name__ == '__main__':
    X = load_data("movies_dataset.csv")
    # X_partial = X[["vote_count", "vote_average"]]
    new_features = np.array([["bradd", (1, 3, 5), 3],
                             ["angelina", (2, 3, 6, 8), 4],
                             ["jake", (4,), 1]])
    remove_rare_columns(new_features, 50)

