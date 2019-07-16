#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import lightgbm as lgb
from model import Model
from utils import save_prediction

def mse(labels, preds):
    score = mean_squared_error(y_true=labels, y_pred=preds)
    return 'mse', score, False

class LGB_Model(Model):
    def __init__(self):
        self.lgb_model = lgb.LGBMRegressor(boosting_type="gbdt", num_leaves=7, reg_alpha=0, reg_lambda=0.1,
            subsample=0.8, colsample_bytree=0.8, subsample_freq=1, min_child_samples=10, metric="None",
            learning_rate=0.1, n_estimators=300, random_state=2019)

    def fit_eval(self, X_train, y_train, X_eval, y_eval):
        eval_set = [(X_eval, y_eval)]
        self.lgb_model.fit(X_train, y_train, eval_set=eval_set, eval_metric=mse, verbose=50, early_stopping_rounds=100)

    def fit(self, X_train, y_train):
        self.lgb_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.lgb_model.predict(X_test)

    def get_submission(self, X_test):
        y_pred = self.predict(X_test)
        save_prediction(y_pred)


if __name__ == "__main__":
    from train import train_eval, generate_submission
    lgb_model = LGB_Model()
    train_eval(lgb_model)
    generate_submission(lgb_model)
