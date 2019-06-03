#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from utils import save_prediction

from ensemble import Ensemble
from lgb import LGB_Model
from xgb import XGB_Model
from etr import ETR_Model

with open('./data/zhengqi_train.txt') as file:
    dataset = pd.read_csv(file, sep='\t')

with open('./data/zhengqi_test.txt') as file:
    test_dataset = pd.read_csv(file, sep='\t')

train_dataset   = dataset.sample(frac=0.8, random_state=2019)
eval_dataset    = dataset.drop(train_dataset.index)

train_labels    = train_dataset.pop('target')
eval_labels     = eval_dataset.pop('target')

X_train = train_dataset.values
X_eval  = eval_dataset.values

y_train = train_labels.values
y_eval  = eval_labels.values

X_test  = test_dataset.values

scaler  = MinMaxScaler()
scaler.fit(np.concatenate((X_train, X_eval), axis=0))
X_train_scaled  = scaler.transform(X_train)
X_eval_scaled   = scaler.transform(X_eval)

model_list = [('LGB', LGB_Model()), ('XGB', XGB_Model()), ('ETR', ETR_Model())]
ens = Ensemble(model_list)
ens.fit_eval(X_train_scaled, y_train, X_eval_scaled, y_eval)

# X_train = np.concatenate((X_train, X_eval), axis=0)
# y_train = np.concatenate((y_train, y_eval), axis=0)

# scaler  = MinMaxScaler()
# scaler.fit(np.concatenate((X_train, X_test), axis=0))
# X_train = scaler.transform(X_train)
# X_test  = scaler.transform(X_test)

# ens.fit(X_train, y_train)
# ens.get_submission(X_test)
