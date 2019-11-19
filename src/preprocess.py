#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


def load_train_data():
    df = pd.read_csv('../data/zhengqi_train.txt', sep='\t')

    X_train = df.drop(columns=['target'])
    y_train = df['target']

    return X_train, y_train


def load_test_data():
    df = pd.read_csv('../data/zhengqi_test.txt', sep='\t')
    X_test = df

    return X_test


def feature_preprocess(X):
    X = X.drop(['V5', 'V9', 'V11', 'V17', 'V22', 'V28'], axis=1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X['V0'] = X['V0'].apply(lambda x: math.exp(x))
    X['V1'] = X['V1'].apply(lambda x: math.exp(x))
    X['V6'] = X['V6'].apply(lambda x: math.exp(x))
    X['V7'] = X['V7'].apply(lambda x: math.exp(x))
    X['V8'] = X['V8'].apply(lambda x: math.exp(x))
    X['V30'] = np.log1p(X['V30'])

    X = pd.DataFrame(preprocessing.scale(X), columns=X.columns)
    return X


def target_preprocess(y):
    return y


def get_data():
    X_train, y_train = load_train_data()
    X_test = load_test_data()

    all_data = pd.concat([X_train, X_test])
    all_data = feature_preprocess(all_data)
    X_train = all_data.iloc[0: X_train.shape[0]]
    X_test = all_data.iloc[X_train.shape[0]:]
    y_train = target_preprocess(y_train)

    return X_train, y_train, X_test
