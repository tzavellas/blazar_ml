#!/usr/bin/env python3
import argparse
import common
import json
import numpy as np
import os
import sys
from tensorflow import keras
import dnn
import rnn


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
        rnd_search_cv = {'dnn': dnn.regress_dnn(output_shape),
                         'avg': dnn.regress_dnn_avg(output_shape),
                         'concat': dnn.regress_dnn_concat(output_shape),
                         'rnn': rnn.regress_simple_rnn(output_shape),
                         'lstm': rnn.regress_lstm(output_shape),
                         'gru': rnn.regress_gru(output_shape)}
        
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