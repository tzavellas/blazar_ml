import keras_tuner as kt
import sys
import argparse
import common
import json
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import rnn
import dnn


if __name__ == "__main__":
    filename = os.path.basename(__file__).split('.')[0]

    parser = argparse.ArgumentParser(
        description='Loads a dataset and performs hyperparameter tuning.')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Config file.')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        sys.exit(1)

    np.set_printoptions(precision=4, suppress=True)

    with open(args.config) as config:
        config = json.loads(config.read())

        if config['dataset']['path'] == 'ENV_VARIABLE_PLACEHOLDER':
            env_var_value = os.getenv('HEA_DATASET_PATH')
            if env_var_value:
                if os.path.exists(env_var_value):
                    config['dataset']['path'] = env_var_value
                else:
                    print(f'HEA_DATASET_PATH={env_var_value} does not exist')
                    sys.exit(1)
            else:
                print('Environment variable HEA_DATASET_PATH is not set!')
                sys.exit(1)

        dataset = config['dataset']
        hyper_parameters = config['hyper_parameters']
        train_parameters = config['train_parameters']
        paths = config['paths']

        working_dir = os.path.abspath(paths.get('working_dir', filename))

        logs = os.path.join(working_dir, paths.get('logs', 'logs'))
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

        dataset_path = dataset['path']
        n_features = dataset['inputs']
        test_ratio = train_parameters['test_ratio']
        train_full, test = common.load_data(dataset_path,
                                            n_features,
                                            test_ratio)

        # Normalize training set
        train_full[1] = common.normalize_clip(train_full[1])
        n_labels = train_full[1].shape[1]

        hyper = {
            'dnn': common.Tuner(dnn.build_model, n_features, n_labels, hyper_parameters),
            'avg': common.Tuner(dnn.build_model_avg, n_features, n_labels, hyper_parameters),
            'concat': common.Tuner(dnn.build_model_concat, n_features, n_labels, hyper_parameters),
            'rnn': common.Tuner(rnn.build_simple_rnn, n_features, n_labels, hyper_parameters),
            'lstm': common.Tuner(rnn.build_lstm, n_features, n_labels, hyper_parameters),
            'gru': common.Tuner(rnn.build_gru, n_features, n_labels, hyper_parameters)
        }

        hypermodel = hyper[train_parameters['architecture']]
        samples = train_parameters['samples']
        overwrite = paths.get('overwrite', True)
        if hyper_parameters['tuner'] == 'random_search':
            tuner = kt.RandomSearch(hypermodel,
                                    objective='val_loss',
                                    max_trials=samples,
                                    seed=42,
                                    directory=working_dir,
                                    project_name='trials',
                                    overwrite=overwrite)
        elif hyper_parameters['tuner'] == 'grid_search':
            tuner = kt.GridSearch(hypermodel,
                                  objective='val_loss',
                                  max_trials=samples,
                                  seed=42,
                                  directory=working_dir,
                                  project_name='trials',
                                  overwrite=overwrite)
        elif hyper_parameters['tuner'] == 'bayensian_optimization':
            tuner = kt.BayesianOptimization(hypermodel,
                                            objective='val_loss',
                                            max_trials=samples,
                                            seed=42,
                                            directory=working_dir,
                                            project_name='trials',
                                            overwrite=overwrite)
        tuner.search_space_summary()

        if overwrite:
            hp = kt.HyperParameters()
            hp.Choice('batch_size', values=train_parameters['batch_size'])
            epochs = train_parameters['epochs']
            tuner.search(
                train_full[0],
                train_full[1],
                batch_size=hp.get('batch_size'),
                epochs=epochs,
                validation_split=train_parameters['validation_ratio'],
                callbacks=[
                    tf.keras.callbacks.TensorBoard(
                        log_dir=logs,
                        update_freq='epoch',
                        histogram_freq=0),
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=int(
                            epochs / 10))],
                use_multiprocessing=True)
        else:
            tuner.reload()

        tuner.results_summary(num_trials=3)

        hyper_params = tuner.get_best_hyperparameters(num_trials=1)
        if hyper_params:
            best_hps = hyper_params[0]
            report = os.path.join(working_dir, paths['report'])
            with open(report, 'w') as f:
                f.write(f'Hidden layers: {best_hps.get("hidden")}\n')
                f.write(f'Neurons : {best_hps.get("neurons")}\n')
                f.write(f'learning rate: {best_hps.get("learning_rate")}\n')
                f.write(f'Batch size: {best_hps.get("batch_size")}')
        else:
            print('Empty hyper parameters')
