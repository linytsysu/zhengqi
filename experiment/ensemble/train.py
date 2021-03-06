#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from utils import save_prediction
from autoencoder import generate_new_feature

from model import Model

def train_eval(model):
    if not isinstance(model, Model):
        return
    with open('../../data/zhengqi_train.txt') as file:
        dataset = pd.read_csv(file, sep='\t')

    dataset.drop(labels=['V5', 'V17', 'V22'], axis=1, inplace=True)

    train_dataset   = dataset.sample(frac=0.8, random_state=2019)
    eval_dataset    = dataset.drop(train_dataset.index)

    train_labels    = train_dataset.pop('target')
    eval_labels     = eval_dataset.pop('target')

    X_train = train_dataset.values
    X_eval  = eval_dataset.values

    y_train = train_labels.values
    y_eval  = eval_labels.values

    X_train, X_eval = generate_new_feature(X_train, X_eval)

    model.fit_eval(X_train, y_train, X_eval, y_eval)


def generate_submission(model):
    if not isinstance(model, Model):
        return

    with open('../../data/zhengqi_train.txt') as file:
        train_dataset = pd.read_csv(file, sep='\t')

    with open('../../data/zhengqi_test.txt') as file:
        test_dataset = pd.read_csv(file, sep='\t')

    train_dataset.drop(labels=['V5', 'V17', 'V22'], axis=1, inplace=True)
    test_dataset.drop(labels=['V5', 'V17', 'V22'], axis=1, inplace=True)

    train_labels = train_dataset.pop('target')

    X_train = train_dataset.values
    y_train = train_labels.values

    X_test = test_dataset.values

    X_train, X_test = generate_new_feature(X_train, X_test)

    model.fit(X_train, y_train)
    model.get_submission(X_test)


if __name__ == "__main__":
    from ensemble import Ensemble
    from lgb import LGB_Model
    from xgb import XGB_Model
    from etr import ETR_Model
    from huber import Huber_Model

    model_list = [('LGB', LGB_Model()), ('XGB', XGB_Model()), ('ETR', ETR_Model()), ('HUBER', Huber_Model())]
    ens = Ensemble(model_list)
    train_eval(ens)
    generate_submission(ens)

