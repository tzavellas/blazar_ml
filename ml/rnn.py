#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasRegressor
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV


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


def regress_simple_rnn(output_shape):
    keras_reg = KerasRegressor(model=build_simple_rnn,
                               model__meta={'n_outputs_expected_': output_shape},
                               optimizer=keras.optimizers.Adam,
                               loss=keras.losses.MeanSquaredLogarithmicError,
                               metrics=[keras.metrics.MeanSquaredLogarithmicError,
                                        keras.metrics.MeanSquaredError])

    param_distribs = {
        "model__n_hidden": [2, 3, 4, 5],
        "model__n_neurons": np.arange(10, 2000),
        "optimizer__learning_rate": reciprocal(1e-5, 1e-2),
    }

    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=20, cv=3)

    return rnd_search_cv


def regress_lstm(output_shape):
    keras_reg = KerasRegressor(model=build_lstm,
                               model__meta={'n_outputs_expected_': output_shape},
                               optimizer=keras.optimizers.Adam,
                               loss=keras.losses.MeanSquaredLogarithmicError,
                               metrics=[keras.metrics.MeanSquaredLogarithmicError,
                                        keras.metrics.MeanSquaredError])

    param_distribs = {
        "model__n_hidden": [2, 3, 4, 5],
        "model__n_neurons": np.arange(10, 2000),
        "optimizer__learning_rate": reciprocal(1e-5, 1e-2),
    }

    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=20, cv=3)

    return rnd_search_cv


def regress_gru(output_shape):
    keras_reg = KerasRegressor(model=build_gru,
                               model__meta={'n_outputs_expected_': output_shape},
                               optimizer=keras.optimizers.Adam,
                               loss=keras.losses.MeanSquaredLogarithmicError,
                               metrics=[keras.metrics.MeanSquaredLogarithmicError,
                                        keras.metrics.MeanSquaredError])

    param_distribs = {
        "model__n_hidden": [2, 3, 4, 5],
        "model__n_neurons": np.arange(10, 2000),
        "optimizer__learning_rate": reciprocal(1e-5, 1e-2),
    }

    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=20, cv=3)

    return rnd_search_cv