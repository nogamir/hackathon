import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import math

split_factor = 0.9
alphas = [10**(-5)*(10**x) for x in range(10)]


class LinearRegressionModel:
    def __init__(self, alpha=0):
        self.model = sklearn.linear_model.Lasso(alpha=alpha)
        self.X = None
        self.y = None

    def train(self, X, y):
        self.X = X
        self.y = y
        self.model.fit(self.X, self.y)

    def predict(self, X):
        return self.model.predict(X)

    def mse(self, y, prediction):
        return mean_squared_error(y, prediction)

    def score(self, X, y):
        return self.model.score(X, y)


def data_process(train_set):
    return train_set[["budget", "vote_count"]],\
           train_set["revenue"], \
           train_set["vote_average"]


def split_train_and_test(data, split_factor):
    train_set = data.sample(frac=split_factor)
    test_set = data.drop(train_set.index)
    return train_set, test_set


def train_and_plot_score(train_y, test_y, name):
    percent_start = 11
    test_score = np.empty(100-percent_start+1)
    for p in range(percent_start, 101):
        model = LinearRegressionModel()
        train_subset = train_design_matrix.head(math.ceil(train_design_matrix.shape[0] * (p / 100)))
        train_subset_response = train_y.head(math.ceil(train_y.shape[0] * (p / 100)))
        model.train(train_subset, train_subset_response)
        test_score[p - percent_start] = model.score(test_design_matrix, test_y)
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(np.arange(test_score.shape[0]) + percent_start, test_score, 'o-', label="score")
    ax1.set_title(f"Score of {name} by Percentage of Training Data")
    ax1.set_xlabel("Percentage of Training Data")
    ax1.set_ylabel(f"Score of {name}")
    ax1.set_yscale('log')
    plt.legend()
    plt.show()


def train_and_plot_mse(train_y, test_y, name):
    percent_start = 5
    test_error = np.empty(100-percent_start+1)
    #train_error = np.empty(100-percent_start+1)
    for p in range(percent_start, 101):
        model = LinearRegressionModel()
        train_subset = train_design_matrix.head(math.ceil(train_design_matrix.shape[0] * (p / 100)))
        train_subset_response = train_y.head(math.ceil(train_y.shape[0] * (p / 100)))
        model.train(train_subset, train_subset_response)
        test_prediction = model.predict(test_design_matrix)
        #train_prediction = model.predict(model.X)
        test_error[p - percent_start] = model.mse(test_y, test_prediction)
        #train_error[p - percent_start] = model.mse(model.y, train_prediction)
    fig = plt.figure()
    ax1 = fig.add_subplot()
    #ax.plot(np.arange(train_error.shape[0]) + percent_start, train_error, 'o-', label="train")
    ax1.plot(np.arange(test_error.shape[0]) + percent_start, test_error, 'o-', label="mse")
    ax1.set_title(f"MSE of {name} by Percentage of Training Data")
    ax1.set_xlabel("Percentage of Training Data")
    ax1.set_ylabel(f"MSE of {name}")
    ax1.set_yscale('log')
    plt.legend()
    plt.show()


def train_and_plot_mse_alpha(train_y, test_y, name):
    test_error = np.empty(10)
    for j in range(1, 101):
        train_set, test_set = split_train_and_test(data, split_factor)
        train_design_matrix, train_revenue_y, train_vote_y = \
            data_process(train_set)
        test_design_matrix, test_revenue_y, test_vote_y = data_process(
            test_set)
        for i in range(len(alphas)):
            model = LinearRegressionModel(alpha=alphas[i])
            model.train(train_design_matrix, train_y)
            test_prediction = model.predict(test_design_matrix)
            test_error[i] = (test_error[i] + model.mse(test_y, test_prediction))/j
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(np.arange(test_error.shape[0]), test_error, 'o-', label="mse")
    ax1.set_title(f"MSE of {name} by Reg Paramteer")
    ax1.set_xlabel("Reg Parameter")
    ax1.set_ylabel(f"MSE of {name}")
    ax1.set_yscale('log')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv("movies_dataset.csv").dropna()
    train_set, test_set = split_train_and_test(data, split_factor)
    train_design_matrix, train_revenue_y, train_vote_y = \
        data_process(train_set)
    test_design_matrix, test_revenue_y, test_vote_y = data_process(test_set)
    train_and_plot_score(train_revenue_y, test_revenue_y, "Revenue")
    train_and_plot_mse(train_revenue_y, test_revenue_y, "Revenue")
    train_and_plot_score(train_vote_y, test_vote_y, "Vote Average")
    train_and_plot_mse(train_vote_y, test_vote_y, "Vote Average")
    #train_and_plot_mse_alpha(train_revenue_y, test_revenue_y, "Revenue")
    #train_and_plot_mse_alpha(train_vote_y, test_vote_y, "Vote Average")


