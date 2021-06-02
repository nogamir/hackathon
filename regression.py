import numpy as np
import pandas as pd
import sklearn.linear_model


class LinearRegressionModel:
    def __init__(self, data):
        self.model = sklearn.linear_model.LinearRegression
        self.data = pd.read_csv(data)

    def load_data(self):
        house_sales_df = load_data("kc_house_data.csv")


