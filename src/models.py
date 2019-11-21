#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import keras
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, LinearRegression
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.pipeline import make_pipeline


def build_nn():
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=128, activation='linear', input_dim=18))
    model.add(keras.layers.Dense(units=32, activation='linear'))
    model.add(keras.layers.Dense(units=8, activation='linear'))
    model.add(keras.layers.Dense(units=1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model


def build_model():
    seed = 2018

    svr = make_pipeline(SVR(kernel='linear'))
    line = make_pipeline(LinearRegression())
    lasso = make_pipeline(Lasso(alpha =0.0005, random_state=seed))
    ENet = make_pipeline(ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=seed))
    KRR1 = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    KRR2 = KernelRidge(alpha=1.5, kernel='linear', degree=2, coef0=2.5)
    lgbm = LGBMRegressor(learning_rate=0.01, n_estimators=500, num_leaves=31)
    xgb = XGBRegressor(booster='gbtree',colsample_bytree=0.8, gamma=0.1,
                        learning_rate=0.02, max_depth=5,
                        n_estimators=500,min_child_weight=0.8,
                        reg_alpha=0, reg_lambda=1,
                        subsample=0.8, silent=1,
                        random_state=seed, nthread = 2)
    nn = KerasRegressor(build_fn=build_nn, nb_epoch=500, batch_size=32, verbose=2)

    return svr, line, lasso, ENet, KRR1, KRR2, lgbm, xgb, nn


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
