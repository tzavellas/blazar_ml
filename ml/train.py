#!/usr/bin/env python3
import common
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
import rnn


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)


if __name__ == "__main__":
    
    np.set_printoptions(precision=4, suppress=True)

    dataset_path = sys.argv[1]

    train_full, test = common.load_data(dataset_path, 0.2) # returns train and test sets

    model = rnn.build_model(n_hidden=3, n_neurons=68, learning_rate=0.001288946565028537)
    # model = rnn.build_model_lstm(n_hidden=3, n_neurons=100)


    model.summary()

    # early_stop = keras.callbacks.EarlyStopping(monitor='root_mean_squared_error', patience=10)
    history = model.fit(train_full[0], train_full[1], epochs=150, validation_split=.2,
    #                     # callbacks=[early_stop]
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
