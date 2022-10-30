#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras


def build_model(n_hidden=3, n_neurons=68, learning_rate=0.001288946565028537, input_shape=(6, 1)):
    model = keras.models.Sequential()
    model.add(keras.layers.SimpleRNN(n_neurons, input_shape=input_shape, return_sequences=True, dtype=tf.float64))
    for layer in range(n_hidden):
        model.add(keras.layers.SimpleRNN(n_neurons, return_sequences=True))
    model.add(keras.layers.SimpleRNN(n_neurons))
    model.add(keras.layers.Dense(500))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.RootMeanSquaredError()],
                  optimizer=optimizer)
    return model


def build_model_lstm(n_hidden=2, n_neurons=10, learning_rate=1e-3, input_shape=(6, 1)):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(n_neurons, input_shape=input_shape, return_sequences=True, dtype=tf.float64))
    for layer in range(n_hidden):
        model.add(keras.layers.LSTM(n_neurons, return_sequences=True))
    model.add(keras.layers.LSTM(n_neurons))
    model.add(keras.layers.Dense(500))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.RootMeanSquaredError()],
                  optimizer=optimizer)
    return model