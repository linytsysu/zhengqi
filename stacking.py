#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from mlxtend.regressor import StackingRegressor
from sklearn.decomposition import PCA

lr = LinearRegression()
svr = SVR()
rf = RandomForestRegressor(n_estimators=100)
abr = AdaBoostRegressor(n_estimators=100)
gbr = GradientBoostingRegressor(n_estimators=100)

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

pca = PCA(n_components=0.95)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

stregr = StackingRegressor(regressors=[lr, svr, rf, abr, gbr], meta_regressor=gbr)
stregr.fit(X_train, y_train)
y_pred = stregr.predict(X_test)
print(mean_squared_error(y_test, y_pred))
