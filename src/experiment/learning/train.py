#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
            label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
            label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()

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

def main():
    with open('./data/zhengqi_train.txt') as file:
        dataset = pd.read_csv(file, sep='\t')

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_labels = train_dataset.pop('target')
    test_labels = test_dataset.pop('target')

    model = build_model(input_shape=[len(train_dataset.keys())])

    EPOCHS = 200

    history = model.fit(
        train_dataset, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0)

    plot_history(history)

    loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)
    print(loss, mae, mse)

if __name__ == "__main__":
    main()
