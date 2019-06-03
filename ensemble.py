#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from utils import save_prediction

class Ensemble:
    def __init__(self, model_list):
        self.model_list = model_list
        self.models = []
        self.ens_model = RandomForestRegressor(n_estimators=100, random_state=2019)

    def fit_eval(self, X_train, y_train, X_eval, y_eval):
        stacked_train_preds = []
        stacked_test_preds  = []
        for model_name, model in self.model_list:
            print('------------ %s ------------'%(model_name))
            model.fit_eval(X_train, y_train, X_eval, y_eval)
            print('------------ End ------------\n')
            stacked_train_preds.append(model.predict(X_train))
            stacked_test_preds.append(model.predict(X_eval))
        stacked_train_preds = np.array(stacked_train_preds).transpose()
        stacked_test_preds  = np.array(stacked_test_preds).transpose()
        self.ens_model.fit(stacked_train_preds, y_train)
        y_pred = self.ens_model.predict(stacked_test_preds)
        print('MSE:', mean_squared_error(y_pred=y_pred, y_true=y_eval))

    def fit(self, X_train, y_train):
        stacked_preds = []
        for _, model in self.model_list:
            model.fit(X_train, y_train)
            stacked_preds.append(model.predict(X_train))
        stacked_preds = np.array(stacked_preds).transpose()
        self.ens_model.fit(stacked_preds, y_train)

    def predict(self, X_test):
        stacked_preds = []
        for _, model in self.model_list:
            stacked_preds.append(model.predict(X_test))
        stacked_preds = np.array(stacked_preds).transpose()
        return self.ens_model.predict(stacked_preds)

    def get_submission(self, X_test):
        y_pred = self.predict(X_test)
        save_prediction(y_pred)
