#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf

with open('./data/zhengqi_train.txt') as file:
    train_dataset = pd.read_csv(file, sep='\t')

with open('./data/zhengqi_test.txt') as file:
    test_dataset = pd.read_csv(file, sep='\t')

train_labels = train_dataset.pop('target')

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.layers.Dense(128, activation=tf.nn.relu, input_shape=input_shape),
        tf.layers.Dense(128, activation=tf.nn.relu),
        tf.layers.Dropout(rate=0.2),
        tf.layers.Dense(128, activation=tf.nn.relu),
        tf.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.00001)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model(input_shape=[len(train_dataset.keys())])

EPOCHS = 200

history = model.fit(
    train_dataset, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0)

prediction = model.predict(test_dataset).flatten()
pd.DataFrame(prediction).to_csv('prediction.txt', index=False, header=False)
