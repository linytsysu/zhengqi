#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

with open('../../data/zhengqi_train.txt') as file:
    dataset = pd.read_csv(file, sep='\t')

train_dataset   = dataset
train_labels    = train_dataset.pop('target')

X_train = train_dataset.values
y_train = train_labels.values

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

model_input = tf.keras.Input(shape=(38,))

split_1 = tf.slice(model_input, [0, 0], [-1, 7])
split_2 = tf.slice(model_input, [0, 7], [-1, 7])
split_3 = tf.slice(model_input, [0, 14], [-1, 7])
split_4 = tf.slice(model_input, [0, 21], [-1, 7])
split_5 = tf.slice(model_input, [0, 28], [-1, 7])
split_6 = tf.slice(model_input, [0, 35], [-1, 3])

hidden_1 = tf.keras.layers.Dense(7, activation='relu')(split_1)
hidden_2 = tf.keras.layers.Dense(7, activation='relu')(split_2)
hidden_3 = tf.keras.layers.Dense(7, activation='relu')(split_3)
hidden_4 = tf.keras.layers.Dense(7, activation='relu')(split_4)
hidden_5 = tf.keras.layers.Dense(7, activation='relu')(split_5)
hidden_6 = tf.keras.layers.Dense(3, activation='relu')(split_6)

merge = tf.keras.layers.concatenate(inputs=[hidden_1, hidden_2, hidden_3, hidden_4, hidden_5, hidden_6], axis=1)

dropout = tf.keras.layers.Dropout(0.04)(merge)
layer_1 = tf.keras.layers.Dense(50, activation='relu')(dropout)
layer_2 = tf.keras.layers.Dense(30, activation='relu')(layer_1)
layer_3 = tf.keras.layers.Dense(20, activation='relu')(layer_2)
model_output  = tf.keras.layers.Dense(1)(layer_3)

model = tf.keras.Model(inputs=model_input, outputs=model_output)
model.compile(optimizer='adam', loss='mse')

model.fit(x=X_train, y=y_train, batch_size=32, epochs=100, verbose=1, validation_split=0.2)
