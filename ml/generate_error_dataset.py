#!/usr/bin/env python3
import argparse
import common
import numpy as np
import os
import pandas as pd
import sys
from tensorflow import keras



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Loads a dataset and plots actual and predicted curves of each test case.')
    parser.add_argument('-w', '--working-dir', default='plots', type=str,
                        help='Root path where the individual plots will be saved. Default is "plots".')
    parser.add_argument('-d', '--dataset', type=str,
                        help='The path of the dataset.')
    parser.add_argument('-m', '--models', type=str, default='models.txt',
                        help='File containing the list of h5 models to plot. Default is models.txt.')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as arg_e:
        print('parse_args: {}'.format(arg_e), file=sys.stderr)
        sys.exit(1)

    dataset = args.dataset
    working_dir = os.path.abspath(args.working_dir)
    models = args.models

    train_set, _ = common.load_data(dataset, 0) # test_set is empty because ratio is 0

    with open(args.models) as f:
        model_files = [line.rstrip('\n') for line in f]
        models = [keras.models.load_model(m) for m in model_files]
    
    predictions = common.calculate_predictions(models, train_set)
    
    error_metric = common.calculate_error(train_set[1],
                                          predictions,
                                          common.kolmogorov_smirnov_error).T
    
    error_dataset = np.concatenate((train_set[0], error_metric), axis=1)
    df = pd.DataFrame(error_dataset)
    
    df.to_csv(os.path.join(working_dir, 'error_dataset.csv'), header=False, index=True)