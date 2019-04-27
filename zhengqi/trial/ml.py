#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def train():
    with open('./data/zhengqi_train.txt') as file:
        dataset = pd.read_csv(file, sep='\t')

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_labels = train_dataset.pop('target')
    test_labels = test_dataset.pop('target')

    train_dataset.drop(["V5", "V9", "V11", "V17", "V22", "V28"], axis=1, inplace=True)
    test_dataset.drop(["V5", "V9", "V11", "V17", "V22", "V28"], axis=1, inplace=True)

    X_train = train_dataset.values
    X_test = test_dataset.values
    
    y_train = train_labels.values
    y_test = test_labels.values

    poly = PolynomialFeatures(2)
    poly.fit(X_train)
    X_train = poly.transform(X_train)
    X_test = poly.transform(X_test)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = GradientBoostingRegressor(n_estimators=400, random_state=1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(mean_squared_error(y_test, y_pred))

def main():
    with open('./data/zhengqi_train.txt') as file:
        train_dataset = pd.read_csv(file, sep='\t')

    with open('./data/zhengqi_test.txt') as file:
        test_dataset = pd.read_csv(file, sep='\t')

    train_dataset.drop(["V5", "V9", "V11", "V17", "V22", "V28"], axis=1, inplace=True)
    test_dataset.drop(["V5", "V9", "V11", "V17", "V22", "V28"], axis=1, inplace=True)

    train_labels = train_dataset.pop('target')

    X_train = train_dataset.values
    y_train = train_labels.values

    X_test = test_dataset.values

    poly = PolynomialFeatures(2)
    poly.fit(X_train)
    X_train = poly.transform(X_train)
    X_test = poly.transform(X_test)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = GradientBoostingRegressor(n_estimators=400)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    pd.DataFrame(y_pred).to_csv('prediction.txt', index=False, header=False)

if __name__ == "__main__":
    main()

