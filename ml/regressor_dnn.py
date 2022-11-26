#!/usr/bin/env python3
import argparse
import common
import json
import numpy as np
import os
from scikeras.wrappers import KerasRegressor
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
import sys
from tensorflow import keras
import dnn


def regress_dnn(output_shape):
    keras_reg = KerasRegressor(model=dnn.build_model,
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
    keras_reg = KerasRegressor(model=dnn.build_model_avg,
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
    keras_reg = KerasRegressor(model=dnn.build_model_concat,
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


if __name__ == "__main__":
    filename = os.path.basename(__file__).split('.')[0]

    parser = argparse.ArgumentParser(
        description='Loads a dataset and trains a DNN.')
    parser.add_argument('-c', '--config', type=str, required=True, 
                        help='Config file.')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        sys.exit(1)

    np.set_printoptions(precision=4, suppress=True)

    with open(args.config) as config:
        config = json.loads(config.read())

        dataset = config['dataset']
        architecture = config['architecture']
        train = config['train']
        working_dir = os.path.abspath(config.get('working_dir', filename))

        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

        dataset_path = dataset['path']
        test_ratio = dataset.get('test', 0.05)
        legacy = dataset.get('legacy', False)
    
        train_full, test = common.load_data(dataset_path, test_ratio, legacy=legacy) # returns train and test sets
        
        output_shape = architecture['outputs']
        rnd_search_cv = {'dnn': regress_dnn(output_shape),
                         'avg': regress_dnn_avg(output_shape),
                         'concat': regress_dnn_concat(output_shape)}
        
        choice = rnd_search_cv[architecture['type']]

        logs = os.path.join(working_dir, train.get('logs', 'logs'))
        validation_ratio = train.get('validation', .2)
        
        choice.fit(train_full[0], train_full[1], epochs=train['epochs'], 
                   validation_split=validation_ratio,
                   callbacks=[keras.callbacks.TensorBoard(logs, update_freq='epoch'),
                              keras.callbacks.EarlyStopping(monitor='loss', patience=5)])

        report = os.path.join(working_dir, train['output'])
        with open(report, 'w') as f:
            f.write('best parameters: {}\n\n'.format(choice.best_params_))
            f.write('best score: {}\n\n'.format(choice.best_score_))
            f.write(str(choice.cv_results_))