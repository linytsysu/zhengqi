#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

with open('./data/zhengqi_train.txt') as file:
    dataset = pd.read_csv(file, sep='\t')

num_cut=[0.042 + 0.431 * i for i in range(-31, 4)]
group_name=[float('{0:b}'.format(i)) for i in range(1, 35)]
print(pd.cut(dataset["V9"], num_cut, labels=group_name))
