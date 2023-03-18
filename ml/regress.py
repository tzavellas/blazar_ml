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

        dataset = config['dataset']
        architecture = config['architecture']
        train = config['train']
        hyper_params = config['hyper_parameters']
        paths = config['paths']
        working_dir = os.path.abspath(paths.get('working_dir', filename))

        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

        dataset_path = dataset['path']
        test_ratio = dataset['test']
        features = architecture['inputs']
        train_full, test = common.load_data(dataset_path,
                                            features,
                                            test_ratio)  # returns train and test sets

        train_full[1] = common.normalize_clip(train_full[1])

        labels = train_full[1].shape[1]

        rnd_search_cv = {
            # 'dnn': dnn.regress_dnn(output_shape, train_params, hyper_params),
            # 'avg': dnn.regress_dnn_avg(output_shape, train_params, hyper_params),
            # 'concat': dnn.regress_dnn_concat(output_shape, train_params, hyper_params),
            'rnn': rnn.regress_simple_rnn(features, labels, train, hyper_params),
            'lstm': rnn.regress_lstm(features, labels, train, hyper_params),
            'gru': rnn.regress_gru(features, labels, train, hyper_params)
        }

        choice = rnd_search_cv[architecture['type']]

        logs = os.path.join(working_dir, paths.get('logs', 'logs'))

        choice.fit(train_full[0], train_full[1],
                   epochs=train['epochs'],
                   validation_split=train.get('validation', .2),
                   callbacks=[tf.keras.callbacks.TensorBoard(logs),
                              # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
                              ])

        report = os.path.join(working_dir, paths.get('report', 'report.txt'))
        with open(report, 'w') as f:
            f.write('best parameters: {}\n\n'.format(choice.best_params_))
            f.write('best score: {}\n\n'.format(choice.best_score_))
            f.write(str(choice.cv_results_))
