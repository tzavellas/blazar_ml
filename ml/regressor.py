#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
import sys
import tensorflow as tf
from tensorflow import keras
import train


def build_model_dnn(n_hidden=4, n_neurons=1000, learning_rate=1e-3, input_shape=[6]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(500))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=keras.losses.MeanSquaredLogarithmicError(),
                  # metrics=[keras.optimizers.RootMeanSquaredError()],
                  optimizer=optimizer)
    return model


if __name__ == "__main__":
    
    np.set_printoptions(precision=4, suppress=True)

    dataset_path = sys.argv[1]

    train_full, test = train.load_data(dataset_path, 0.2) # returns train and test sets
    
    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model_dnn)
    
    # keras_reg.fit(train_full[0], train_full[1], epochs=150, validation_split=.2,)
    param_distribs = {
        "n_hidden": [2,3,4,5],
        "n_neurons": np.arange(500,2000),
        "learning_rate": reciprocal(3e-4, 3e-2),
    }
    
    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
    rnd_search_cv.fit(*train_full, epochs=150, validation_split=.2, 
                      callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    
    model = rnd_search_cv.best_estimator_.model
    
    if len(sys.argv) > 2:
        working_dir = sys.argv[2]
    else:
        working_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(working_dir, 'hea_dnn.h5')
    model.save(save_path)