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

        # Read config dictionaries
        dataset = config['dataset']
        architecture = config['architecture']
        train = config['train']
        working_dir = os.path.abspath(config.get('working_dir', filename))

        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

        # Read dataset configuration
        dataset_path = dataset['path']
        test_ratio = dataset.get('test', 0.05)
        legacy = dataset.get('legacy', False)
        train_full, test = common.load_data(dataset_path,
                                            test_ratio,
                                            legacy=legacy) # returns train and test sets

        # Read dataset configuration
        meta = common.get_meta(architecture)
        hidden = architecture['hidden']
        neurons = architecture['neurons']
        base = architecture.get('base', 2)

        # Build all types of models
        models = {'dnn': dnn.build_model(meta, hidden, neurons),
                  'avg': dnn.build_model_avg(meta, base, hidden, neurons),
                  'concat': dnn.build_model_concat(meta, base, hidden, neurons),
                  'rnn': rnn.build_simple_rnn(meta, hidden, neurons),
                  'lstm': rnn.build_lstm(meta, hidden, neurons),
                  'gru': rnn.build_gru(meta, hidden, neurons)}

        # Choose a type and compile it
        model = models[architecture['type']]
        model = common.compile_model(model, params={'learning_rate': train['learning_rate']})
        model.summary()

        logs = os.path.join(working_dir, train.get('logs', 'logs'))

        # Train the model
        history = model.fit(train_full[0], train_full[1],
                            epochs=train['epochs'],
                            validation_split=train.get('validation', .2),
                            callbacks=[keras.callbacks.TensorBoard(logs, update_freq='epoch')])
        # Evaluate the model
        mse_test = model.evaluate(*test)
        print('MSE test: {}'.format(mse_test))

        # Save the model
        save_path = os.path.join(working_dir, train['output'])
        model.save(save_path)