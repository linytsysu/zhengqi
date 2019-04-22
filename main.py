#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import HuberRegressor, Lasso, RidgeCV
from sklearn.decomposition import PCA

import lightgbm as lgb
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

def xgb_train(X_train, y_train):
    xgtrain = xgb.DMatrix(X_train, y_train)
    params = {
        'eta': 0.03,
        'max_depth': 3,
        'subsample': 0.9,
        'colsample_bylevel': 0.9,
        'eval_metric': 'rmse',
    }
    bst = xgb.train(params, xgtrain, 200)
    return bst

def lgb_train(X_train, y_train):
    train_data = lgb.Dataset(data=X_train, label=y_train)
    param = {
        'num_leaves': 15,
        'num_trees': 100,
        'metric': 'rmse',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'objective': 'huber'
    }
    bst = lgb.train(param, train_data)
    return bst

def rf_train(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    return rf

def xgb_predict(bst, X_test):
    xgtest = xgb.DMatrix(X_test)
    return bst.predict(xgtest)

def lgb_predit(bst, X_test):
    return bst.predict(X_test)

def save_prediction(y_pred):
    pd.DataFrame(y_pred).to_csv('prediction.txt', index=False, header=False)

def remove_outlier(X_train, y_train):
    outlier_dector = OneClassSVM(nu=0.05)
    outlier_dector.fit(X_train)
    pred = outlier_dector.predict(X_train)
    outliers_index = np.where([pred == -1])[1]
    X_train = np.delete(X_train, outliers_index, axis=0)
    y_train = np.delete(y_train, outliers_index)
    return X_train, y_train

def generate_new_feature(X_train, y_train, X_test):
    bst = xgb_train(X_train, y_train)
    xgtrain = xgb.DMatrix(X_train, y_train)
    xgtest = xgb.DMatrix(X_test)

    X_train_new = bst.predict(xgtrain, pred_leaf=True)
    X_test_new = bst.predict(xgtest, pred_leaf=True)

    encoder = OneHotEncoder()
    encoder.fit(X_train_new)
    X_train_new = encoder.transform(X_train_new)
    X_test_new = encoder.transform(X_test_new)

    X_train_new = hstack((X_train_new, X_train))
    X_test_new = hstack((X_test_new, X_test))

    return X_train_new, X_test_new


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

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    lbst = lgb_train(X_train, y_train)
    y_pred_1 = lgb_predit(lbst, X_test)
    print(mean_squared_error(y_pred_1, y_test))

    # lasso = Lasso(alpha=0.01)
    # lasso.fit(X_train, y_train)
    # y_pred_2 = lasso.predict(X_test)
    # print(mean_squared_error(y_pred_2, y_test))

    lr = HuberRegressor()
    lr.fit(X_train, y_train)
    y_pred_3 = lr.predict(X_test)
    print(mean_squared_error(y_pred_3, y_test))

    xbst = xgb_train(X_train, y_train)
    y_pred_4 = xgb_predict(xbst, X_test)
    print(mean_squared_error(y_pred_4, y_test))

    y_pred = np.mean([y_pred_1, y_pred_3, y_pred_4], axis=0)
    print(mean_squared_error(y_pred, y_test))


def main():
    with open('./data/zhengqi_train.txt') as file:
        train_dataset = pd.read_csv(file, sep='\t')

    with open('./data/zhengqi_test.txt') as file:
        test_dataset = pd.read_csv(file, sep='\t')

    train_labels = train_dataset.pop('target')

    X_train = train_dataset.values
    X_test = test_dataset.values

    y_train = train_labels.values

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    lbst = lgb_train(X_train, y_train)
    y_pred_1 = lgb_predit(lbst, X_test)

    # lasso = Lasso(alpha=0.01)
    # lasso.fit(X_train, y_train)
    # y_pred_2 = lasso.predict(X_test)

    lr = HuberRegressor()
    lr.fit(X_train, y_train)
    y_pred_3 = lr.predict(X_test)

    xbst = xgb_train(X_train, y_train)
    y_pred_4 = xgb_predict(xbst, X_test)

    y_pred = np.mean([y_pred_1, y_pred_3, y_pred_4], axis=0)
    save_prediction(y_pred)

if __name__ == "__main__":
    main()
