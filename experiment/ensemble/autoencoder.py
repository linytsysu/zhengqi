#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

tf.set_random_seed(2019)

def generate_new_feature(X_train, X_test):
    X       = np.concatenate((X_train, X_test), axis=0)
    scaler  = MinMaxScaler()
    X       = scaler.fit_transform(X)

    input_size  = X.shape[1]
    hidden_size = 64
    code_size   = 128
    output_size = X.shape[1]

    x           = tf.keras.Input(shape=(input_size,))
    hidden_1    = tf.keras.layers.Dense(hidden_size, activation='relu')(x)
    h           = tf.keras.layers.Dense(code_size, activation='relu')(hidden_1)
    hidden_2    = tf.keras.layers.Dense(hidden_size, activation='relu')(h)
    r           = tf.keras.layers.Dense(output_size, activation='sigmoid')(hidden_2)

    autoencoder = tf.keras.Model(inputs=x, outputs=r)
    autoencoder.compile(optimizer='adam', loss='mse')

    epochs = 80
    batch_size = 16
    autoencoder.fit(x=X, y=X, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.2)

    conv_encoder    = tf.keras.Model(x, h)

    train_hidden_feature    = conv_encoder.predict(scaler.transform(X_train))
    test_hidden_feature     = conv_encoder.predict(scaler.transform(X_test))
    return train_hidden_feature, test_hidden_feature
