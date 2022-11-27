#!/usr/bin/env python3
import argparse
import common
import json
import numpy as np
import os
import pandas as pd
import sys
from tensorflow import keras


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Loads a dataset and plots actual and predicted curves of each test case.')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Config file.')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        sys.exit(1)

    with open(args.config) as config:
        config = json.loads(config.read())

        dataset = config['dataset']
        working_dir = os.path.abspath(config['working_dir'])

        if not os.path.exists(working_dir):
            os.mkdir(working_dir)
    
        train_set, _ = common.load_data(dataset['path'], dataset.get('test', 0)) # test_set is empty because ratio is 0
    
        model_files = [os.path.basename(file) for file in config['models']]
        models = [keras.models.load_model(m) for m in config['models']]

        predictions = common.calculate_predictions(models, train_set)

        error_metric = common.calculate_error(train_set[1],
                                              predictions,
                                              common.kolmogorov_smirnov_error).T

        error_dataset = np.concatenate((train_set[0], error_metric), axis=1)
        df = pd.DataFrame(error_dataset)

        df.to_csv(os.path.join(working_dir, config['output']), header=False, index=True)
