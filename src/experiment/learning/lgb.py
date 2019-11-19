#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def train():
    with open('./data/zhengqi_train.txt') as file:
        dataset = pd.read_csv(file, sep='\t')

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_labels = train_dataset.pop('target')
    test_labels = test_dataset.pop('target')

    X_train = train_dataset.values
    X_test = test_dataset.values

    y_train = train_labels.values
    y_test = test_labels.values

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    train_data = lgb.Dataset(data=X_train, label=y_train)

    param = {
        'num_leaves': 31,
        'num_trees': 100,
        'metric': 'rmse',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'objective': 'huber'
    }
    bst = lgb.train(param, train_data, 500)
    y_pred = bst.predict(X_test)

    print(mean_squared_error(y_test, y_pred))

if __name__ == "__main__":
    train()
