#!/usr/bin/env python3
import argparse
import common
import json
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
import dnn
import rnn


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


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
        paths = config['paths']
        working_dir = os.path.abspath(paths.get('working_dir', filename))

        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

        # Read dataset configuration
        dataset_path = dataset['path']
        test_ratio = dataset['test']
        features = architecture['inputs']
        train_full, test = common.load_data(dataset_path,
                                            features,
                                            test_ratio)  # returns train and test sets

        train_full[1] = common.normalize_clip(train_full[1])

        labels = train_full[1].shape[1]

        # Read dataset configuration
        hidden = architecture['hidden']
        neurons = architecture['neurons']
        base = architecture.get('base', 2)

        # Build all types of models
        models = {
            'dnn': dnn.build_model(features, labels, hidden, neurons),
            'avg': dnn.build_model_avg(features, labels, base, hidden, neurons),
            'concat': dnn.build_model_concat(features, labels, base, hidden, neurons),
            'rnn': rnn.build_simple_rnn(features, labels, hidden, neurons),
            'lstm': rnn.build_lstm(features, labels, hidden, neurons),
            'gru': rnn.build_gru(features, labels, hidden, neurons)
        }

        # Choose a type and compile it
        model = models[architecture['type']]
        model = common.compile_model(
            model, params={
                'learning_rate': train['learning_rate']})
        model.summary()

        logs = os.path.join(working_dir, paths.get('logs', 'logs'))
        backup = os.path.join(working_dir, paths.get('backup', 'backup'))

        # Train the model
        history = model.fit(train_full[0], train_full[1],
                            epochs=train['epochs'],
                            validation_split=train.get('validation', .2),
                            callbacks=[keras.callbacks.TensorBoard(logs),
                                       keras.callbacks.BackupAndRestore(
                                           backup),
                                       # keras.callbacks.LearningRateScheduler(scheduler)
                                       ])
        # Evaluate the model
        mse_test = model.evaluate(*test)
        print('MSE test: {}'.format(mse_test))

        # Save the model
        save_path = os.path.join(working_dir, paths['output'])
        print('Saving model at: {}'.format(save_path))
        model.save(save_path)
