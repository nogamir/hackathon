import numpy as np
import pandas as pd
import sklearn.linear_model

SPLIT_FACTOR = 0.9
# to do: separate responses
#        split data
#

class LinearRegressionModel:
    def __init__(self, data_path):
        self.model = sklearn.linear_model.LinearRegression()
        data = pd.read_csv(data_path)
        train_set, self.test_set = self.split_train_and_test(data)
        self.design_matrix = train_set.drop(["price"], ["vote_average"], axis=1)
        self.vote_label = train_set["vote_average"]
        self.revenue_label = train_set["revenue"]
        self.process_data()

    def process_data(self):
        return

    def train_for_vote(self):
        self.model.fit(self.design_matrix, self.vote_label)

    def train_for_revenue(self):
        self.model.fit(self.design_matrix, self.vote_label)

    def split_train_and_test(self, data):
        train_set = data.sample(frac=SPLIT_FACTOR)
        test_set = data.drop(train_set.index)
        return train_set, test_set


