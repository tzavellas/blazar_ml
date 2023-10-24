import argparse
import common
import json
import numpy as np
import os
import sys
import tensorflow as tf
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

        # Check config dictionary for environment replacement
        try:
            config = common.check_environment(config)
        except ValueError as e:
            print(f'{e}')
            sys.exit(1)

        # Read config dictionaries
        dataset = config['dataset']
        train_parameters = config['train_parameters']
        paths = config['paths']

        working_dir = os.path.abspath(paths.get('working_dir', './train'))
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

        # Read dataset configuration
        dataset_path = dataset['path']
        test_ratio = train_parameters['test_ratio']
        n_features = dataset['inputs']

        try:
            train_full, test = common.load_data(
                dataset_path, n_features, test_ratio)  # returns train and test sets
        except Exception as e:
            print(f'Loading data: {e}')
            sys.exit(1)

        # train_full[1] = common.normalize_clip2(train_full[1])
        # test[1] = common.normalize_clip2(test[1])

        train_full[1] = common.normalize_clip(train_full[1])
        test[1] = common.normalize_clip(test[1])

        n_labels = train_full[1].shape[1]

        # Read dataset configuration
        hidden = train_parameters['hidden']
        neurons = train_parameters['neurons']
        base = train_parameters.get('base', 2)
        name = train_parameters.get('name', None)

        # Build all types of models
        models = {
            'dnn': dnn.build_model(n_features, n_labels, hidden, neurons, name),
            'rnn': rnn.build_simple_rnn(n_features, n_labels, hidden, neurons, name),
            'lstm': rnn.build_lstm(n_features, n_labels, hidden, neurons, name),
            'gru': rnn.build_gru(n_features, n_labels, hidden, neurons, name)
        }

        # Choose a type and compile it
        model = models[train_parameters['architecture']]
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=train_parameters['learning_rate']),
            loss=tf.keras.losses.MeanSquaredLogarithmicError(),
            metrics=tf.keras.metrics.MeanSquaredError())
        model.summary()

        logs = os.path.join(working_dir, f'logs_{name}')
        backup = os.path.join(working_dir, f'backup_{name}')
        # Train the model
        history = model.fit(train_full[0], train_full[1],
                            epochs=train_parameters['epochs'],
                            batch_size=train_parameters['batch_size'],
                            validation_split=train_parameters['validation_ratio'],
                            callbacks=[tf.keras.callbacks.TensorBoard(logs),
                                       tf.keras.callbacks.BackupAndRestore(
                                           backup),
                                       # tf.keras.callbacks.LearningRateScheduler(common.scheduler)
                                       ])
        # Evaluate the model
        mse_test = model.evaluate(*test)
        print(f'MSE test: {mse_test}')

        # Save the model
        save_path = os.path.join(working_dir, paths['output'])
        print(f'Saving model at: {save_path}')
        model.save(save_path)
