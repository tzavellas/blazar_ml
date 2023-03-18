#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasRegressor
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV


def build_simple_rnn(n_features, n_labels, n_hidden=3, n_neurons=100):
    input_shape = n_features, 1
    output_shape = n_labels

    model = keras.models.Sequential()
    model.add(
        keras.layers.SimpleRNN(
            n_neurons,
            input_shape=input_shape,
            return_sequences=True,
            dtype=tf.float64))
    for layer in range(n_hidden):
        model.add(keras.layers.SimpleRNN(n_neurons, return_sequences=True))
    model.add(keras.layers.SimpleRNN(n_neurons))
    model.add(keras.layers.Dense(output_shape))

    return model


def build_lstm(n_features, n_labels, n_hidden=3, n_neurons=100):
    input_shape = n_features, 1
    output_shape = n_labels

    model = keras.models.Sequential()
    model.add(
        keras.layers.LSTM(
            n_neurons,
            input_shape=input_shape,
            return_sequences=True,
            dtype=tf.float64))
    for layer in range(n_hidden):
        model.add(keras.layers.LSTM(n_neurons, return_sequences=True))
    model.add(keras.layers.LSTM(n_neurons))
    model.add(keras.layers.Dense(output_shape, activation='relu'))

    return model


def build_gru(n_features, n_labels, n_hidden=3, n_neurons=100):
    input_shape = n_features, 1
    output_shape = n_labels

    model = keras.models.Sequential()
    model.add(
        keras.layers.GRU(
            n_neurons,
            input_shape=input_shape,
            return_sequences=True,
            dtype=tf.float64))
    for layer in range(n_hidden):
        model.add(keras.layers.GRU(n_neurons, return_sequences=True))
    model.add(keras.layers.GRU(n_neurons))
    model.add(keras.layers.Dense(output_shape))

    return model


def regress_simple_rnn(n_features, n_labels, train_params, hyper_params):
    keras_reg = KerasRegressor(model=build_simple_rnn,
                               model__n_features=n_features,
                               model__n_labels=n_labels,
                               optimizer=keras.optimizers.Adam,
                               loss=keras.losses.MeanSquaredLogarithmicError,
                               metrics=[keras.metrics.MeanSquaredLogarithmicError,
                                        keras.metrics.MeanSquaredError])

    param_distribs = {
        "model__n_hidden": np.arange(*hyper_params['hidden']),
        "model__n_neurons": np.power(2, np.arange(*hyper_params['neuron_exponent'])),
        "optimizer__learning_rate": reciprocal(*hyper_params['learning_rate']),
    }

    rnd_search_cv = RandomizedSearchCV(keras_reg,
                                       param_distribs,
                                       n_iter=train_params.get('samples', 10),
                                       cv=train_params.get('cv', None))

    return rnd_search_cv


def regress_lstm(n_features, n_labels, train_params, hyper_params):
    keras_reg = KerasRegressor(model=build_lstm,
                               model__n_features=n_features,
                               model__n_labels=n_labels,
                               optimizer=keras.optimizers.Adam,
                               loss=keras.losses.MeanSquaredLogarithmicError,
                               metrics=[keras.metrics.MeanSquaredLogarithmicError,
                                        keras.metrics.MeanSquaredError])

    param_distribs = {
        'model__n_hidden': np.arange(*hyper_params['hidden']),
        'model__n_neurons': np.power(2, np.arange(*hyper_params['neuron_exponent'])),
        'optimizer__learning_rate': reciprocal(*hyper_params['learning_rate']),
    }

    rnd_search_cv = RandomizedSearchCV(keras_reg,
                                       param_distribs,
                                       n_iter=train_params.get('samples', 10),
                                       cv=train_params.get('cv', None))

    return rnd_search_cv


def regress_gru(n_features, n_labels, train_params, hyper_params):
    keras_reg = KerasRegressor(model=build_gru,
                               model__n_features=n_features,
                               model__n_labels=n_labels,
                               optimizer=keras.optimizers.Adam,
                               loss=keras.losses.MeanSquaredLogarithmicError,
                               metrics=[keras.metrics.MeanSquaredLogarithmicError,
                                        keras.metrics.MeanSquaredError])

    param_distribs = {
        "model__n_hidden": np.arange(*hyper_params['hidden']),
        "model__n_neurons": np.power(2, np.arange(*hyper_params['neuron_exponent'])),
        "optimizer__learning_rate": reciprocal(*hyper_params['learning_rate']),
    }

    rnd_search_cv = RandomizedSearchCV(keras_reg,
                                       param_distribs,
                                       n_iter=train_params.get('samples', 10),
                                       cv=train_params.get('cv', None))

    return rnd_search_cv
