#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from model import Model
from utils import save_prediction

def mse(preds, dtrain):
    labels = dtrain.get_label()
    score = mean_squared_error(y_true=labels, y_pred=preds)
    return 'mse', score

class XGB_Model(Model):
    def __init__(self):
        self.xgb_model = xgb.XGBRegressor(reg_alpha=0, reg_lambda=0.01, n_estimators=1000, random_state=2019,
            subsample=0.8, colsample_bytree=0.8, learning_rate=0.1)

    def fit_eval(self, X_train, y_train, X_eval, y_eval):
        eval_set = [(X_eval, y_eval)]
        self.xgb_model.fit(X_train, y_train, eval_set=eval_set, eval_metric=mse, verbose=50, early_stopping_rounds=100)

    def fit(self, X_train, y_train):
        self.xgb_model.n_estimators = self.xgb_model.best_iteration
        self.xgb_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.xgb_model.predict(X_test)

    def get_submission(self, X_test):
        y_pred = self.predict(X_test)
        save_prediction(y_pred)



if __name__ == "__main__":
    from train import train_eval, generate_submission
    xgb_model = XGB_Model()
    train_eval(xgb_model)
    generate_submission(xgb_model)
