#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import HuberRegressor
from utils import save_prediction
from model import Model

class Huber_Model(Model):
    def __init__(self):
        self.huber_model = HuberRegressor(epsilon=1., alpha=0.001)

    def fit_eval(self, X_train, y_train, X_eval, y_eval):
        self.huber_model.fit(X_train, y_train)
        y_pred = self.huber_model.predict(X_eval)
        print('MSE: ', mean_squared_error(y_true=y_eval, y_pred=y_pred))

    def fit(self, X_train, y_train):
        self.huber_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.huber_model.predict(X_test)

    def get_submission(self, X_test):
        y_pred = self.predict(X_test)
        save_prediction(y_pred)

if __name__ == "__main__":
    from train import train_eval, generate_submission
    huber_model = Huber_Model()
    train_eval(huber_model)
    generate_submission(huber_model)
