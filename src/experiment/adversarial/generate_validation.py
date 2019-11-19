#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import lightgbm as lgb
import matplotlib.pyplot as plt

train_dataset = pd.read_csv('../../data/zhengqi_train.txt', sep='\t')
test_dataset = pd.read_csv('../../data/zhengqi_test.txt', sep='\t')

target = train_dataset.pop('target')

train_dataset['is_train'] = 1
test_dataset['is_train'] = 0

# train_dataset.drop(labels=['V5', 'V14', 'V22', 'V9', 'V7', 'V21', 'V23', 'V19', 'V17', 'V11', 'V13', 'V28', 'V3', 'V25', 'V26', 'V20', 'V16', 'V35'], axis=1, inplace=True)
# test_dataset.drop(labels=['V5', 'V14', 'V22', 'V9', 'V7', 'V21', 'V23', 'V19', 'V17', 'V11', 'V13', 'V28', 'V3', 'V25', 'V26', 'V20', 'V16', 'V35'], axis=1, inplace=True)

dataset = pd.concat([train_dataset, test_dataset])
dataset = dataset.sample(frac=1.0)

y = dataset.pop('is_train').values
X = dataset.values

model = lgb.LGBMClassifier()
model.fit(X, y)


y_pred = model.predict_proba(train_dataset.drop(['is_train'], axis=1).values)[:, 1]
idx = np.argsort(y_pred)
for i in range(100):
    print(target[idx[i]])

# y_pred = model.predict_proba(test_dataset.drop(['is_train'], axis=1).values)[:, 1]
# y_pred = np.sort(y_pred)
# for i in range(100):
#     print(y_pred[-i - 1])
# print(accuracy_score(y_true=train_dataset['is_train'].values, y_pred=y_pred))

# y_pred = model.predict(test_dataset.drop(['is_train'], axis=1).values)
# print(accuracy_score(y_true=test_dataset['is_train'].values, y_pred=y_pred))

