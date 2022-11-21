#!/usr/bin/env python3
import common
import numpy as np
import os
import sys
from tensorflow import keras
import dnn


if __name__ == "__main__":
    
    np.set_printoptions(precision=4, suppress=True)

    dataset_path = '/home/tzavellas/hea_ml_generated_files/datasets/dataset_10k/normalized.csv'

    train_full, test = common.load_data(dataset_path, 0.2) # returns train and test sets

    model = dnn.build_model()

    model.summary()

    history = model.fit(train_full[0], train_full[1], epochs=600, validation_split=.2,
                        callbacks=[keras.callbacks.TensorBoard('./logs/dnn', update_freq='epoch')])

    mse_test = model.evaluate(*test)

    if len(sys.argv) > 2:
        working_dir = sys.argv[2]
    else:
        working_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(working_dir, 'hea_dnn.h5')
    model.save(save_path)
