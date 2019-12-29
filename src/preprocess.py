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
import xgboost as xgb
import talib
import tsfresh as tsf


def load_train_data():
    df = pd.read_csv('../data/zhengqi_train.txt', sep='\t')

    X_train = df.drop(columns=['target'])
    y_train = df['target']

    return X_train, y_train


def load_test_data():
    df = pd.read_csv('../data/zhengqi_test.txt', sep='\t')
    X_test = df

    return X_test


def feature_test(X_train, y_train, X_test):
    column_names = X_train.columns
    X_train['is_train'] = 1
    X_test['is_train'] = 0

    xgb_params = {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1,
                  'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0,
                  'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 6,
                  'min_child_weight': 1, 'missing': None, 'n_estimators': 500,
                  'n_jobs': 1, 'nthread': -1, 'objective': 'binary:logistic',
                  'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1,
                  'seed': 917, 'silent': False, 'subsample': 1, 'verbosity': 1, 'tree_method': 'auto'}
    for name in column_names:
        all_data = pd.concat([X_train[[name,'is_train']], X_test[[name,'is_train']]])
        xgtrain = xgb.DMatrix(all_data[[name]], label=all_data['is_train'])
        cvresult_shift = xgb.cv(xgb_params, xgtrain, num_boost_round=100, nfold=5,
                                metrics='auc', seed=1, early_stopping_rounds=5)
        if cvresult_shift['test-auc-mean'][cvresult_shift.shape[0]-1] > 0.8:
            print(name, cvresult_shift['test-auc-mean'][cvresult_shift.shape[0]-1])


def feature_preprocess(X):
    X = X.drop(['V5', 'V9', 'V11', 'V17', 'V22', 'V28'], axis=1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X['V0'] = X['V0'].apply(lambda x: math.exp(x))
    X['V1'] = X['V1'].apply(lambda x: math.exp(x))
    X['V6'] = X['V6'].apply(lambda x: math.exp(x))
    X['V7'] = X['V7'].apply(lambda x: math.exp(x))
    X['V8'] = X['V8'].apply(lambda x: math.exp(x))

    X = pd.DataFrame(preprocessing.scale(X), columns=X.columns)
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
    print(feat_scored_headnum)
    X_train = X_train[X_train.columns[X_train.columns.isin(feat_scored_headnum)]]
    X_test = X_test[X_test.columns[X_test.columns.isin(feat_scored_headnum)]]

    return X_train, X_test


def time_series_feature_generation(X, y):
    # timeperiod = 1
    # if timeperiod > 0:
    #     X['SHIFT_V0'] = X['V0'].shift(periods=timeperiod).values
    #     X['SHIFT_V1'] = X['V1'].shift(periods=timeperiod).values
    #     X['SHIFT_V2'] = X['V2'].shift(periods=timeperiod).values
    #     X['SHIFT_V31'] = X['V31'].shift(periods=timeperiod).values

    # X['NEW_2'] = np.average([X['V0'].values, X['V1'].values, X['V2'].values, X['V8'].values, X['V31'].values], axis=0, weights=[5, 4, 3, 2, 1])
    # X['NEW_3'] = np.average([X['V0'].values, X['V1'].values, X['V2'].values, X['V31'].values], axis=0, weights=[1, 2, 3, 4])
    # X['NEW_4'] = np.average([X['V0'].values, X['V1'].values, X['V2'].values, X['V31'].values], axis=0, weights=[4, 3, 2, 1])

    for feature_name in X.columns:
        new_feature = []
        for i in range(X[feature_name].shape[0]):
            if i < 1:
                new_feature.append(np.nan)
            else:
                new_feature.append((X[feature_name].values[i] - X[feature_name].values[i-1]))
        X['CHANGES_%s'%(feature_name)] = new_feature

    X = pd.DataFrame(preprocessing.scale(X), columns=X.columns)

    return X


def get_data():
    X_train, y_train = load_train_data()
    X_test = load_test_data()

    # feature_test(X_train, y_train, X_test)

    all_data = pd.concat([X_train, X_test])
    all_data = feature_preprocess(all_data)

    X_train = pd.DataFrame(all_data.iloc[0: X_train.shape[0]].values, columns=all_data.columns)
    X_test = pd.DataFrame(all_data.iloc[X_train.shape[0]:].values, columns=all_data.columns)

    # X_train = X_train.copy().iloc[1:, :]
    # y_train = pd.Series(y_train.values[1:])

    X_train, X_test = feature_selection(X_train, y_train, X_test)

    all_data = pd.concat([X_train, X_test])
    all_data = time_series_feature_generation(all_data, y_train)
    X_train = pd.DataFrame(all_data.iloc[0: X_train.shape[0]].values, columns=all_data.columns)
    X_test = pd.DataFrame(all_data.iloc[X_train.shape[0]:].values, columns=all_data.columns)

    X_train = pd.DataFrame(X_train.iloc[1000:].values, columns=X_train.columns)
    y_train = pd.Series(y_train.values[1000:])

    return X_train, y_train, X_test
