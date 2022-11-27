#!/usr/bin/env python3
import argparse
import common
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tensorflow import keras
# from sklearn import metrics
# from dtaidistance import dtw


def mean_error_ranking(error, model_ids):
    """
    Calculates the mean error and std per row. Returns a ranking of the mean
    error as well as a list of tuples (mean, std)

    Parameters
    ----------
    error : nd.array
        The error metric for each case and each model [num_models x num_cases].
    model_ids : list
        List of model ids.

    Returns
    -------
    tuple(list, list)
        The first list is the ranking of the mean error, the second is a list
        of (mean, std).

    """
    m_id = np.array(model_ids)
    mm = np.mean(error, axis=1)
    ss = np.std(error, axis=1)
    argsort = np.argsort(mm)    
    return m_id[argsort].tolist(), list(zip(mm,ss))


def mean_ranking(error, model_ids):
    """
    Calculates the ranking of each case and returns the mean ranking, the mean
    value of each model and the array of all rankings

    Parameters
    ----------
    error : nd.array
        The error metric for each case and each model [num_models x num_cases].
    model_ids : list
        List of model ids.

    Returns
    -------
    list
        Mean rank.
    mean : nd.array
        Mean values of each model.
    rankings : nd.array
        The rank of each case.

    """
    argsort = np.argsort(error, axis=0)
    rankings = argsort + 1
    mean = np.mean(rankings, axis=1)
    m_id = np.array(model_ids)
    argsort = np.argsort(mean)
    m_id[argsort].tolist()
    return m_id[argsort].tolist(), mean, rankings


def plot_grouped_barchart(rankings, model_ids, out_dir):
    n = rankings.shape[0]
    counts = np.zeros((n, n))
    for i, rank in enumerate(rankings):
        for j, model in enumerate(model_ids):
            counts[i, j] = np.count_nonzero(rankings[i, :] == (j + 1))
    fig, ax = plt.subplots()
    x = np.arange(len(model_ids), dtype=np.int16)
    width = 0.15
    pos = x - 3*width/2
    for i, c in enumerate(counts):
        ax.bar(pos, c, width, label=model_ids[i])
        pos = pos + width
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Rank')
    ax.legend()
    
    # Creates x-axis string labels, eg 1, 2, 3, ...
    labels = []
    for i in range(n):
        labels.append('{}'.format(i + 1))
    ax.set_xticks(x, labels)
    
    fig_path = os.path.join(out_dir, 'ranking_bar_chart.svg')
    plt.savefig(fig_path, format='svg')
    plt.close(fig)
    

def plot(y, mark, lab):
    plt.plot(y, marker=mark, label=lab, markersize=2, linestyle='')
    plt.ylim(-30, 0)
    # plt.xlim(300, 450)


def plot_matrix(data, name, labels, out_dir, symbols='ox+|_', scale='linear'):
    fig = plt.figure()
    for i, y in enumerate(data):
        plt.plot(y, marker=symbols[i], linestyle='',  label=labels[i])
        plt.xlabel('case')
        plt.ylabel(name)
        plt.yscale(scale)
        plt.legend()
    fig_path = os.path.join(out_dir, '{}.svg'.format(name))
    plt.savefig(fig_path, format='svg')
    plt.close(fig)

    
def plot_all_cases(y, prediction, labels, out_dir, symbols='ox+|_'):
    y_pred = common.de_normalize(prediction)
    y_d = common.de_normalize(y)    # index 1 means the spectrum values
    for i, y_i in enumerate(y_d):
        fig = plt.figure(i)
        plot(y_i, symbols[0], 'actual')
        for j, y_p in enumerate(y_pred):
            plot(y_p[i], symbols[j + 1], labels[j])
        plt.legend()
        fig_path = os.path.join(out_dir, '{}.svg'.format(i))
        plt.savefig(fig_path, format='svg')
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Loads a dataset and plots actual and predicted curves of each test case.')
    parser.add_argument('-w', '--working-dir', default='plots', type=str,
                        help='Root path where the individual plots will be saved. Default is "plots".')
    parser.add_argument('-d', '--dataset', type=str,
                        help='The path of the dataset.')
    parser.add_argument('-m', '--models', type=str, default='models.txt',
                        help='File containing the list of h5 models to plot. Default is models.txt.')
    parser.add_argument('-a', '--all', action='store_true',
                        help='Plots the prediction along with the actual spectrum.')
    parser.add_argument('-r', '--rankings', type=bool, default=True,
                        help='Plots a grouped barchart of the rankings. Default is False.')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as arg_e:
        print('parse_args: {}'.format(arg_e), file=sys.stderr)
        sys.exit(1)

    dataset = args.dataset
    working_dir = os.path.abspath(args.working_dir)
    models = args.models
    plot_rankings = args.rankings
    plot_all = args.all

    train_set, _ = common.load_data(dataset, 0) # test_set is empty because ratio is 0

    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    with open(models) as f:
        model_files = [line.rstrip('\n') for line in f]
    
    models = [keras.models.load_model(m) for m in model_files]

    prediction = common.calculate_predictions(models, train_set)

    error_metric = common.calculate_error(train_set[1],
                                          prediction,
                                          common.kolmogorov_smirnov_error)
    # alternative error functions: dtw.distance, common.integral_error, metrics.mean_squared_error
    
    rank_of_mean, stats = mean_error_ranking(error_metric, model_files)
    
    mean_rank, mean_val, rankings = mean_ranking(error_metric, model_files)
        
    plot_matrix(error_metric, 'error', model_files, working_dir, scale='log')

    if plot_all == True:
        plot_all_cases(train_set[1], prediction, model_files, working_dir)

    if plot_rankings == True:
        plot_grouped_barchart(rankings, model_files, working_dir)
        # plot_matrix(rankings, 'ranking', model_files, working_dir)
