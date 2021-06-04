import time
import LoadData
import pandas as pd
import sklearn.linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xg
import matplotlib.pyplot as plt
import numpy as np
import math

SPLIT_FACTOR = 0.90


class LinearRegressionModel:
    def __init__(self, mats):
        self.mats = mats
        self.vote_model = xg.XGBRegressor(objective ='reg:linear', n_estimators = 10, seed = 123)
        self.revenue_model = xg.XGBRegressor(objective='reg:linear', n_estimators=10, seed=123)
        self.X = None

    def train(self, X, y, mode):
        X = LoadData.modify_data(X, self.mats, mode)
        X = X.drop('status', axis='columns')
        self.X = X
        if mode == 0:
            self.vote_model.fit(X, y)
        else:
            self.revenue_model.fit(X, y)



    def predict(self, X, mode):
        not_released_idx = np.where(X.status != 'Released')
        X = LoadData.modify_test(X, self.mats, mode)
        # collection_idx = np.where(X.belongs_to_collection != 'Released')
        X = X.drop('status', axis='columns')
        if mode == 0:
            y_hat = self.vote_model.predict(X)
        else:
            y_hat = self.revenue_model.predict(X)
        if len(not_released_idx[0]) > 0:
            np.put(y_hat, not_released_idx, 0)
        return y_hat

    def mse(self, y, prediction):
        return mean_squared_error(y, prediction)


def data_process(train_set):

    return train_set[["budget"]],\
           train_set["revenue"], \
           train_set["vote_average"]


def split_train_and_test(data, split_factor):
    train_set = data.sample(frac=split_factor)
    test_set = data.drop(train_set.index)
    return train_set, test_set


def train_model(train_X, train_y, model):
    model.train(train_X, train_y)


def train_and_plot_mse(train_X, train_y, test_X, test_y, mats, name, mode):
    percent_start = 5
    run_range = range(percent_start, 101, 10)
    test_error = np.empty(len(run_range))
    # vecs = [90,80,70,60,50,40,30,20]
    # for vec in vecs:
    #     LoadData.set_feature(vec)
    #     new_name = f'{name}_vec={vec}'
    for i in range(len(run_range)):
        train_X_copy, train_y_copy, test_X_copy, test_y_copy = train_X.copy(), train_y.copy(), test_X.copy(), \
                                                               test_y.copy()
        model = LinearRegressionModel(mats)
        ceil = math.ceil(train_X_copy.shape[0] * (run_range[i] / 100))
        train_subset = train_X_copy.head(ceil)
        train_subset_response = train_y_copy.head(ceil)
        model.train(train_subset, train_subset_response, mode)
        test_prediction = model.predict(test_X_copy, mode)
        test_error[i] = np.sqrt(model.mse(test_y_copy, test_prediction))
    plot_mse(run_range, test_error, name, mode)


def plot_mse(run_range, test_error, name, mode):
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(run_range, test_error, 'o-', label="mse")
    ax1.set_title(f"MSE of {name} by Percentage of Training Data")
    ax1.set_xlabel("Percentage of Training Data")
    ax1.set_ylabel(f"MSE of {name}")
    if mode == 1:
        ax1.set_yscale('log')
    plt.legend()
    plt.show()


def drop_cols(df):
    cols = ['budget', 'vote_count', 'runtime', 'score_value_genres',
            'score_value_production_companies', 'score_value_keywords',
            'score_value_cast', 'score_value_crew']
    return df[cols]



if __name__ == '__main__':

    df, mats = LoadData.get_df_and_mats()
    extra_data = pd.read_csv("movies_dataset_part2.csv")

    # train_set, test_set = split_train_and_test(df, SPLIT_FACTOR)
    train_set, test_set = df, extra_data

    train_rev_y = train_set["revenue"]
    train_vote_y = train_set["vote_average"]
    train_X = train_set.drop(['revenue', 'vote_average'], axis='columns')
    test_rev_y = test_set["revenue"]
    test_vote_y = test_set["vote_average"]
    test_X = test_set.drop(['revenue', 'vote_average'], axis='columns')
    train_and_plot_mse(train_X, train_vote_y, test_X, test_vote_y, mats, "Vote Average", 0)
    train_and_plot_mse(train_X, train_rev_y, test_X, test_rev_y, mats, "Revenue", 1)

    # model = LinearRegressionModel(mats)
    # model.train(train_X, train_rev_y)
    # test_prediction = model.predict(test_X)

    # df = pd.read_csv("movies_dataset.csv")
    # vec = np.empty((len(df),))
    # df = pd.concat((df, pd.DataFrame(vec), axis=1))
    # print("hi")






