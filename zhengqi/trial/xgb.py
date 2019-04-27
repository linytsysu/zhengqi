#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def train():
    with open('./data/zhengqi_train.txt') as file:
        dataset = pd.read_csv(file, sep='\t')

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_labels = train_dataset.pop('target')
    test_labels = test_dataset.pop('target')

    train_dataset['new1'] = train_dataset['V1'] * train_dataset['V2']
    test_dataset['new1'] = test_dataset['V1'] * test_dataset['V2']

    X_train = train_dataset.values
    X_test = test_dataset.values

    y_train = train_labels.values
    y_test = test_labels.values

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    xgtrain = xgb.DMatrix(X_train, y_train)
    xgtest = xgb.DMatrix(X_test)

    params = {
        'eta': 0.01,
        'max_depth': 6,
        'subsample': 0.9,
        'colsample_bylevel': 0.6,
        'eval_metric': 'rmse',
    }
    bst = xgb.train(params, xgtrain, 1000)
    y_pred = bst.predict(xgtest)

    print(mean_squared_error(y_test, y_pred))

if __name__ == "__main__":
    train()
