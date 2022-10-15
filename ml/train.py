#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
from tensorflow import keras



def load_data(path, test_ratio=0.2, random_state=42, drop_na=True):
    '''
    Loads a dataset in csv format and returns a training set and a test set
    
    Parameters
    ----------
    path : string
        Path to csv dataset.
    test_ratio : float, optional
        Percentage of the dataset to use as test set. The default is 0.2.
    random_state : int, optional
        Seed for random number generation. The default is 42.

    Returns
    -------
    Tuple of tuples of np.array
        The training set and the test set.

    '''
    raw_dataset = pd.read_csv(path, header=None, index_col=0, na_values='NaN')
    if drop_na:
        dataset = raw_dataset.dropna()
    else:
        dataset = raw_dataset

    train_set = dataset.sample(frac=1-test_ratio, random_state=random_state)
    test_set = dataset.drop(train_set.index)

    n_features = 6

    x_train = train_set.iloc[:, :n_features].to_numpy()
    y_train = train_set.iloc[:, n_features:-1].to_numpy()

    x_test = test_set.iloc[:, :n_features].to_numpy()
    y_test = test_set.iloc[:, n_features:-1].to_numpy()

    return (x_train, y_train), (x_test, y_test)


def split_valid(x_train_full, y_train_full, ratio=0.2):
    '''
    Splits a full training set and returns a validation set and a training set
    
    Parameters
    ----------
    x_train_full : np.array
        The features of the training set.
    y_train_full : np.array
        The labels of the training set.
    ratio : float, optional
        Percentage of the full training set to use as validation set. The default is 0.2.

    Returns
    -------
    Tuple of tuples of np.array
        The validation set and the training set.

    '''
    n = int(len(x_train_full)*ratio)
    
    x_valid, x_train = x_train_full[:n], x_train_full[n:]
    y_valid, y_train = y_train_full[:n], y_train_full[n:]
    return (x_valid, y_valid), (x_train, y_train)
    

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)


def build_model_dnn(n_hidden=4, n_neurons=1000, learning_rate=1e-3, input_shape=[6]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(500))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=keras.losses.MeanSquaredLogarithmicError(),
                  metrics=[keras.optimizers.RootMeanSquaredError()],
                  optimizer=optimizer)
    return model


if __name__ == "__main__":
    
    np.set_printoptions(precision=4, suppress=True)

    dataset_path = sys.argv[1]

    train_full, test = load_data(dataset_path, 0.2) # returns train and test sets

    # valid, train = split_valid(*train_full) # splits to valid and train sets
    # exponential_decay_fn = exponential_decay(lr0=0.01, s=50)
    # lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)

    model = keras.models.Sequential([
        keras.Input(shape=(6,)),
        keras.layers.Dense(1000, activation="relu"),
        keras.layers.Dense(1000, activation="relu"),
        keras.layers.Dense(1000, activation="relu"),
        keras.layers.Dense(1000, activation="relu"),
        keras.layers.Dense(500)
        ])
    model.compile(loss=keras.losses.MeanSquaredLogarithmicError(),
                  metrics=[keras.metrics.RootMeanSquaredError()],
                  optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  )

    history = model.fit(train_full[0], train_full[1], epochs=150, 
                        validation_split=.2,
                        # callbacks=[lr_scheduler]
                        )

    mse_test = model.evaluate(*test)

    print('MSE test {}'.format(mse_test))
    plot_loss(history)

    if len(sys.argv) > 2:
        working_dir = sys.argv[2]
    else:
        working_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(working_dir, 'hea.h5')
    model.save(save_path)
