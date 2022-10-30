#!/usr/bin/env python3
import common
import numpy as np
import os
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
import sys
from tensorflow import keras
import rnn


def regress_rnn():
    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(rnn.build_model)

    param_distribs = {
        "n_hidden": [2, 3, 4, 5],
        "n_neurons": np.arange(10, 2000),
        "learning_rate": reciprocal(1e-5, 1e-2),
    }

    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=20, cv=3)

    return rnd_search_cv


def regress_lstm():
    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(rnn.build_model_lstm)

    param_distribs = {
        "n_hidden": [2, 3, 4, 5],
        "n_neurons": np.arange(10, 2000),
        "learning_rate": reciprocal(1e-5, 1e-2),
    }

    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=20, cv=3)

    return rnd_search_cv


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    dataset_path = sys.argv[1]

    train_full, test = common.load_data(dataset_path, 0.2) # returns train and test sets

    # rnd_search_cv = regress_rnn()
    rnd_search_cv = regress_lstm()
    # rnd_search_cv = regress_dnn_concat()

    rnd_search_cv.fit(*train_full, epochs=150, validation_split=.2,
                      callbacks=[keras.callbacks.EarlyStopping(patience=10)])

    model = rnd_search_cv.best_estimator_.model

    with open('regress_lstm_report.txt', 'w') as f:
        f.write('best parameters: {}\n\n'.format(rnd_search_cv.best_params_))
        f.write('best score: {}\n\n'.format(rnd_search_cv.best_score_))
        f.write(str(rnd_search_cv.cv_results_))

    if len(sys.argv) > 2:
        working_dir = sys.argv[2]
    else:
        working_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(working_dir, 'hea_lstm.h5')
    model.save(save_path)