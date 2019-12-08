#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


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

    #scaler = MinMaxScaler(feature_range=(0, 1))
    #X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    #X['V0'] = X['V0'].apply(lambda x: math.exp(x))
    #X['V1'] = X['V1'].apply(lambda x: math.exp(x))
    #X['V6'] = X['V6'].apply(lambda x: math.exp(x))
    #X['V7'] = X['V7'].apply(lambda x: math.exp(x))
    #X['V8'] = X['V8'].apply(lambda x: math.exp(x))

    #X['V30'] = np.log1p(X['V30'])

    #X = pd.DataFrame(preprocessing.scale(X), columns=X.columns)
    return X


def feature_selection(X_train, y_train, X_test):
    threshold = 0.85
    vt = VarianceThreshold().fit(X_train)
    feat_var_threshold = X_train.columns[vt.variances_ > threshold * (1 - threshold)]
    X_train = X_train[feat_var_threshold]
    X_test = X_test[feat_var_threshold]

    head_feature_num = 18
    X_scored = SelectKBest(score_func=f_regression, k='all').fit(X_train, y_train)
    feature_scoring = pd.DataFrame({
            'feature': X_train.columns,
            'score': X_scored.scores_
        })
    feature_scoring = feature_scoring.sort_values('score', ascending=False).reset_index(drop=True)
    feat_scored_headnum = feature_scoring.head(head_feature_num)['feature']
    X_train = X_train[X_train.columns[X_train.columns.isin(feat_scored_headnum)]]
    X_test = X_test[X_test.columns[X_test.columns.isin(feat_scored_headnum)]]

    return X_train, X_test


def get_data():
    X_train, y_train = load_train_data()
    X_test = load_test_data()

    all_data = pd.concat([X_train, X_test])
    all_data = feature_preprocess(all_data)

    X_train = all_data.iloc[0: X_train.shape[0]]
    X_test = all_data.iloc[X_train.shape[0]:]

    X_train, X_test = feature_selection(X_train, y_train, X_test)

    return X_train, y_train, X_test
