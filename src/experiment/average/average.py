#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import mean_squared_error
from model import Model
from utils import save_prediction

class Average(Model):
    def __init__(self, model_list):
        self.model_list = model_list

    def fit_eval(self, X_train, y_train, X_eval, y_eval):
        stacked_test_preds = []
        for model_name, model in self.model_list:
            print('------------ %s ------------'%(model_name))
            model.fit_eval(X_train, y_train, X_eval, y_eval)
            print('------------ End ------------\n')
            stacked_test_preds.append(model.predict(X_eval))
        y_pred = np.mean(stacked_test_preds, axis=0)
        print('MSE: ', mean_squared_error(y_pred=y_pred, y_true=y_eval))

    def fit(self, X_train, y_train):
        for _, model in self.model_list:
            model.fit(X_train, y_train)

    def predict(self, X_test):
        stacked_preds = []
        for _, model in self.model_list:
            stacked_preds.append(model.predict(X_test))
        return np.mean(stacked_preds, axis=0)

    def get_submission(self, X_test):
        y_pred = self.predict(X_test)
        save_prediction(y_pred)
