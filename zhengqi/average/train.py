#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from utils import save_prediction

from model import Model

def train_eval(model):
    if not isinstance(model, Model):
        return
    with open('../../data/zhengqi_train.txt') as file:
        dataset = pd.read_csv(file, sep='\t')

    scaler = preprocessing.MinMaxScaler()
    dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)

    dataset.drop(labels=['V5', 'V17', 'V22'], axis=1, inplace=True)

    dataset['V0']   = dataset['V0'].apply(lambda x: math.exp(x))
    dataset['V1']   = dataset['V1'].apply(lambda x: math.exp(x))
    dataset['V6']   = dataset['V6'].apply(lambda x: math.exp(x))
    dataset['V7']   = dataset['V7'].apply(lambda x: math.exp(x))
    dataset['V8']   = dataset['V8'].apply(lambda x: math.exp(x))
    dataset['V30']  = np.log1p(dataset["V30"])

    dataset = pd.DataFrame(preprocessing.scale(dataset), columns=dataset.columns)

    train_dataset   = dataset.sample(frac=0.8, random_state=2019)
    eval_dataset    = dataset.drop(train_dataset.index)

    train_labels    = train_dataset.pop('target')
    eval_labels     = eval_dataset.pop('target')

    X_train = train_dataset.values
    X_eval  = eval_dataset.values

    y_train = train_labels.values
    y_eval  = eval_labels.values

    model.fit_eval(X_train, y_train, X_eval, y_eval)


def generate_submission(model):
    if not isinstance(model, Model):
        return

    with open('../../data/zhengqi_train.txt') as file:
        train_dataset = pd.read_csv(file, sep='\t')

    with open('../../data/zhengqi_test.txt') as file:
        test_dataset = pd.read_csv(file, sep='\t')

    train_labels = train_dataset.pop('target')

    all_dataset = pd.concat([train_dataset, test_dataset])

    scaler = preprocessing.MinMaxScaler()
    all_dataset = pd.DataFrame(scaler.fit_transform(all_dataset), columns=all_dataset.columns)

    all_dataset.drop(labels=['V5', 'V17', 'V22'], axis=1, inplace=True)

    all_dataset['V0']   = all_dataset['V0'].apply(lambda x: math.exp(x))
    all_dataset['V1']   = all_dataset['V1'].apply(lambda x: math.exp(x))
    all_dataset['V6']   = all_dataset['V6'].apply(lambda x: math.exp(x))
    all_dataset['V7']   = all_dataset['V7'].apply(lambda x: math.exp(x))
    all_dataset['V8']   = all_dataset['V8'].apply(lambda x: math.exp(x))
    all_dataset['V30']  = np.log1p(all_dataset["V30"])

    all_dataset = pd.DataFrame(preprocessing.scale(all_dataset), columns=all_dataset.columns)

    X_train = all_dataset[0:len(train_dataset)].values
    X_test  = all_dataset[len(train_dataset):].values
    y_train = train_labels.values

    model.fit(X_train, y_train)
    model.get_submission(X_test)


if __name__ == "__main__":
    from average import Average
    from lgb import LGB_Model
    from etr import ETR_Model
    from huber import Huber_Model

    model_list = [('LGB', LGB_Model()), ('ETR', ETR_Model()), ('HUBER', Huber_Model())]
    ave = Average(model_list)
    train_eval(ave)
    generate_submission(ave)

