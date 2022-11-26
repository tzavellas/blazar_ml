#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras


def build_simple_rnn(meta={}, n_hidden=3, n_neurons=100):
    input_shape = meta.get('n_features_in_', 6)[0], 1
    output_shape = meta.get('n_outputs_expected_', 500)

    model = keras.models.Sequential()
    model.add(keras.layers.SimpleRNN(n_neurons, input_shape=input_shape, return_sequences=True, dtype=tf.float64))
    for layer in range(n_hidden):
        model.add(keras.layers.SimpleRNN(n_neurons, return_sequences=True))
    model.add(keras.layers.SimpleRNN(n_neurons))
    model.add(keras.layers.Dense(output_shape))

    return model


def build_lstm(meta={}, n_hidden=3, n_neurons=100):
    input_shape = meta.get('n_features_in_', 6)[0], 1
    output_shape = meta.get('n_outputs_expected_', 500)

    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(n_neurons, input_shape=input_shape, return_sequences=True, dtype=tf.float64))
    for layer in range(n_hidden):
        model.add(keras.layers.LSTM(n_neurons, return_sequences=True))
    model.add(keras.layers.LSTM(n_neurons))
    model.add(keras.layers.Dense(output_shape))

    return model


def build_gru(meta={}, n_hidden=3, n_neurons=100):
    input_shape = meta.get('n_features_in_', 6)[0], 1
    output_shape = meta.get('n_outputs_expected_', 500)

    model = keras.models.Sequential()
    model.add(keras.layers.GRU(n_neurons, input_shape=input_shape, return_sequences=True, dtype=tf.float64))
    for layer in range(n_hidden):
        model.add(keras.layers.GRU(n_neurons, return_sequences=True))
    model.add(keras.layers.GRU(n_neurons))
    model.add(keras.layers.Dense(output_shape))

    return model