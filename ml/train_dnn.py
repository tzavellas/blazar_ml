#!/usr/bin/env python3
import common
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tensorflow import keras
import dnn


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

    model = dnn.build_model(n_hidden=4, n_neurons=1260, learning_rate=6.7e-3)
    # model = dnn.build_model_avg(n_base=2, n_hidden=4, n_neurons=1260, learning_rate=6.7e-3)
    # model = dnn.build_model_concat(n_base=2, n_hidden=4, n_neurons=1260, learning_rate=6.7e-3)

    # lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
    # early_stop = keras.callbacks.EarlyStopping(monitor='root_mean_squared_error', patience=10)
    history = model.fit(train_full[0], train_full[1], epochs=200, validation_split=.2,
                        # callbacks=[early_stop]
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
