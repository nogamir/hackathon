import sys

sys.path.append("../")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    return df

if __name__ == '__main__':
    path = "movies_dataset.csv"
    X = load_data(path)
    print(X.shape)