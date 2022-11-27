#!/usr/bin/env python3
import numpy as np
from tensorflow import keras
from scikeras.wrappers import KerasRegressor
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV


def base_dense(input_shape, output_shape, hidden_layer_sizes):
    '''
    Creates a model of a dense network, given a set of parameters.

    Parameters
    ----------
    input_shape : list[int]
        The input dimensions.
    output_shape : list[int]
        The output dimensions.
    hidden_layer_sizes : list[int]
        List of integers, representing the number of neurons, of each hidden layer.

    Returns
    -------
    model : tf.keras.Model
        The model.

    '''
    input_layer = keras.layers.Input(shape=input_shape)
    hidden = input_layer
    for n_neurons in hidden_layer_sizes:
        hidden = keras.layers.Dense(n_neurons, activation="relu")(hidden)
    output_layer = keras.layers.Dense(output_shape)(hidden)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def build_model(meta={}, n_hidden=3, n_neurons=1000):
    input_shape = meta.get('n_features_in_', [6])
    output_shape = meta.get('n_outputs_expected_', 500)

    hidden_layer_sizes = [n_neurons for i in range(n_hidden)]
    model = base_dense(input_shape, output_shape, hidden_layer_sizes)

    return model


def build_model_avg(meta={}, n_base=2, n_hidden=3, n_neurons=1000):
    input_shape = meta.get('n_features_in_', [6])
    output_shape= meta.get('n_outputs_expected_', 500)

    hidden_layer_sizes = [n_neurons for i in range(n_hidden)]

    input_layer = keras.layers.Input(shape=input_shape)
    base_out = base_dense(input_shape, output_shape, hidden_layer_sizes)
    layers = [base_out(input_layer) for i in range(n_base)]
    avg = keras.layers.Average()(layers)
    output_layer = keras.layers.Dense(output_shape)(avg)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model


def build_model_concat(meta={}, n_base=2, n_hidden=3, n_neurons=1000):
    input_shape = meta.get('n_features_in_', [6])
    output_shape= meta.get('n_outputs_expected_', 500)

    hidden_layer_sizes = [n_neurons for i in range(n_hidden)]

    input_layer = keras.layers.Input(shape=input_shape)
    base_out = base_dense(input_shape, output_shape, hidden_layer_sizes)
    layers = [base_out(input_layer) for i in range(n_base)]
    merged = keras.layers.concatenate(layers)
    output_layer = keras.layers.Dense(output_shape, activation='relu')(merged)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model


def regress_dnn(output_shape):
    keras_reg = KerasRegressor(model=build_model,
                               model__meta={'n_outputs_expected_': output_shape},
                               optimizer=keras.optimizers.Adam,
                               loss=keras.losses.MeanSquaredLogarithmicError,
                               metrics=[keras.metrics.MeanSquaredLogarithmicError,
                                        keras.metrics.MeanSquaredError])
    param_distribs = {
        "model__n_hidden": [2, 3, 4, 5],
        "model__n_neurons": np.arange(100, 2000),
        "optimizer__learning_rate": reciprocal(1e-5, 1e-2),
    }
    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)

    return rnd_search_cv


def regress_dnn_avg(output_shape):
    keras_reg = KerasRegressor(model=build_model_avg,
                               model__meta={'n_outputs_expected_': output_shape},
                               optimizer=keras.optimizers.Adam,
                               loss=keras.losses.MeanSquaredLogarithmicError,
                               metrics=[keras.metrics.MeanSquaredLogarithmicError,
                                        keras.metrics.MeanSquaredError])
    param_distribs = {
        "model__n_base": [2, 3, 4, 5],
        "model__n_hidden": [2, 3, 4, 5],
        "model__n_neurons": np.arange(500, 2000),
        "optimizer__learning_rate": reciprocal(1e-5, 1e-2),
    }
    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)

    return rnd_search_cv


def regress_dnn_concat(output_shape):
    keras_reg = KerasRegressor(model=build_model_concat,
                               model__meta={'n_outputs_expected_': output_shape},
                               optimizer=keras.optimizers.Adam,
                               loss=keras.losses.MeanSquaredLogarithmicError,
                               metrics=[keras.metrics.MeanSquaredLogarithmicError,
                                        keras.metrics.MeanSquaredError])
    param_distribs = {
        "model__n_base": [2, 3, 4, 5],
        "model__n_hidden": [2, 3, 4, 5],
        "model__n_neurons": np.arange(500, 2000),
        "optimizer__learning_rate": reciprocal(1e-5, 1e-2),
    }
    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)

    return rnd_search_cv