#!/usr/bin/env python3
from tensorflow import keras


def base_dense(n_hidden=4, n_neurons=1000, input_shape=[6]):
    input_ = keras.layers.Input(shape=input_shape)
    hidden = input_
    for layer in range(n_hidden):
        hidden = keras.layers.Dense(n_neurons, activation="relu")(hidden)
    output = keras.layers.Dense(500)(hidden)
    model = keras.Model(inputs=input_, outputs=output)
    return model


def build_model(n_hidden=4, n_neurons=1000, learning_rate=1e-3, input_shape=[6]):
    model = base_dense(n_hidden, n_neurons, input_shape)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=keras.losses.MeanSquaredLogarithmicError(),
                  metrics=[keras.metrics.RootMeanSquaredError()],
                  optimizer=optimizer)
    return model


def build_model_avg(n_base=2, n_hidden=5, n_neurons=634, learning_rate=0.0027031221642068, input_shape=[6]):
    input_ = keras.layers.Input(shape=input_shape)

    base_out = base_dense(n_hidden, n_neurons, input_shape)

    layers = [base_out(input_) for i in range(n_base)]

    avg = keras.layers.Average()(layers)
    out = keras.layers.Dense(500)(avg)

    model = keras.Model(inputs=input_, outputs=out)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss=keras.losses.MeanSquaredLogarithmicError(),
                  metrics=[keras.metrics.RootMeanSquaredError()],
                  optimizer=optimizer)
    return model


def build_model_concat(n_base=2, n_hidden=4, n_neurons=1000, learning_rate=1e-3, input_shape=[6]):
    input_ = keras.layers.Input(shape=input_shape)

    base_out = base_dense(n_hidden, n_neurons, input_shape)

    layers = [base_out(input_) for i in range(n_base)]

    merged = keras.layers.concatenate(layers)
    out = keras.layers.Dense(500, activation='relu')(merged)

    model = keras.Model(inputs=input_, outputs=out)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss=keras.losses.MeanSquaredLogarithmicError(),
                  metrics=[keras.metrics.RootMeanSquaredError()],
                  optimizer=optimizer)
    return model