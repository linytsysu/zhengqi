#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from model import Model
from utils import save_prediction

class ETR_Model(Model):
    def __init__(self):
        self.etr_model = ExtraTreesRegressor(n_estimators=200, random_state=2019)

    def fit_eval(self, X_train, y_train, X_eval, y_eval):
        self.etr_model.fit(X_train, y_train)
        y_pred = self.etr_model.predict(X_eval)
        print('MSE: ', mean_squared_error(y_true=y_eval, y_pred=y_pred))

    def fit(self, X_train, y_train):
        self.etr_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.etr_model.predict(X_test)

    def get_submission(self, X_test):
        y_pred = self.predict(X_test)
        save_prediction(y_pred)


if __name__ == "__main__":
    from train import train_eval, generate_submission
    etr_model = ETR_Model()
    train_eval(etr_model)
    generate_submission(etr_model)

