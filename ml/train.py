import argparse
import common
import json
import os
import shutil
import sys
import tensorflow as tf
import dnn
import rnn


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description='Loads a dataset and trains a Neural Net.')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Config file.')
    return parser.parse_args(argv)


def main(args):
    with open(args.config) as config:
        config = json.loads(config.read())

        # Check config dictionary for environment replacement
        try:
            config = common.check_environment(config)
        except ValueError as e:
            print(f'{e}')
            return 1

        # Read config dictionaries
        dataset = config['dataset']
        train_parameters = config['train_parameters']
        paths = config['paths']

        working_dir = os.path.join(os.path.abspath(paths.get(
            'working_dir', 'train')), train_parameters['architecture'], train_parameters['name'])
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        # Read dataset configuration
        dataset_path = dataset['path']
        test_ratio = train_parameters['test_ratio']
        n_features = dataset['inputs']

        try:
            train_full, test = common.load_data(
                dataset_path, n_features, test_ratio)  # returns train and test sets
        except Exception as e:
            print(f'Loading data: {e}')
            return 1

        # train_full[1] = common.normalize_clip2(train_full[1])
        # test[1] = common.normalize_clip2(test[1])

        train_full[1] = common.normalize_clip(train_full[1])
        test[1] = common.normalize_clip(test[1])

        n_labels = train_full[1].shape[1]

        # Read dataset configuration
        hidden = train_parameters['hidden']
        neurons = train_parameters['neurons']
        name = train_parameters['name']

        # Build all types of models
        models = {
            'dnn': dnn.build_model(n_features, n_labels, hidden, neurons, name),
            'rnn': rnn.build_simple_rnn(n_features, n_labels, hidden, neurons, name),
            'lstm': rnn.build_lstm(n_features, n_labels, hidden, neurons, name),
            'gru': rnn.build_gru(n_features, n_labels, hidden, neurons, name)}
        # Initialize paths
        logs = os.path.join(working_dir, 'logs')
        backup = os.path.join(working_dir, f'backup_{name}')
        save_path = os.path.join(working_dir, f'{name}.h5')

        if os.path.exists(save_path) and (
                paths.get('on_exists', None) is None):
            model = tf.keras.models.load_model(save_path)
        else:
            if os.path.exists(save_path):
                shutil.copy2(save_path, f'{save_path}.old')
            if paths.get('on_exists', 'overwrite') == 'overwrite':
                shutil.rmtree(backup, ignore_errors=True)

            # Choose a type and compile it
            model = models[train_parameters['architecture']]
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=train_parameters['learning_rate']),
                loss=tf.keras.losses.MeanSquaredLogarithmicError(),
                metrics=tf.keras.metrics.MeanSquaredError())
            model.summary()
            # Train the model
            history = model.fit(
                train_full[0],
                train_full[1],
                verbose=2,
                epochs=train_parameters['epochs'],
                batch_size=train_parameters['batch_size'],
                validation_split=train_parameters['validation_ratio'],
                callbacks=[
                    tf.keras.callbacks.TensorBoard(logs),
                    tf.keras.callbacks.BackupAndRestore(backup),
                ])
            # Save the model
            print(f'Saving model at: {save_path}')
            model.save(save_path)

        # Evaluate the model
        if test_ratio > 0:
            eval = model.evaluate(*test)
            print('\nEvaluation')
            for i in range(len(eval)):
                print(f'{model.metrics_names[i]}={eval[i]}')

            if paths.get('report', True):
                report = os.path.join(working_dir, f'report_{name}.txt')
                with open(report, 'w') as f:
                    f.write('Evaluation\n')
                    for i in range(len(eval)):
                        f.write(f'{model.metrics_names[i]}={eval[i]}\n')

    return 0


if __name__ == "__main__":
    try:
        args = parse_arguments(sys.argv[1:])
        ret = main(args)
    except argparse.ArgumentError as e:
        print(e)
    sys.exit(ret)
