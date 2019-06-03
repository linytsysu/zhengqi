#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from utils import save_prediction

def mse(preds, dtrain):
    labels = dtrain.get_label()
    score = mean_squared_error(y_true=labels, y_pred=preds)
    return 'mse', score

class XGB_Model:
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


if __name__ == "__main__":
    with open('./data/zhengqi_train.txt') as file:
        dataset = pd.read_csv(file, sep='\t')

    with open('./data/zhengqi_test.txt') as file:
        test_dataset = pd.read_csv(file, sep='\t')

    train_dataset = dataset.sample(frac=0.8, random_state=2019)
    eval_dataset = dataset.drop(train_dataset.index)

    train_labels = train_dataset.pop('target')
    eval_labels = eval_dataset.pop('target')

    X_train = train_dataset.values
    X_eval = eval_dataset.values

    y_train = train_labels.values
    y_eval = eval_labels.values

    X_test = test_dataset.values

    scaler  = MinMaxScaler()
    scaler.fit(np.concatenate((X_train, X_eval), axis=0))
    X_train_scaled  = scaler.transform(X_train)
    X_eval_scaled   = scaler.transform(X_eval)

    xgb_model = XGB_Model()
    xgb_model.fit_eval(X_train_scaled, y_train, X_eval_scaled, y_eval)

    X_train = np.concatenate((X_train, X_eval), axis=0)
    y_train = np.concatenate((y_train, y_eval), axis=0)

    scaler  = MinMaxScaler()
    scaler.fit(np.concatenate((X_train, X_test), axis=0))
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    save_prediction(y_pred)