import argparse
import common
from dtaidistance import dtw
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from sklearn import metrics
import tensorflow as tf
from astropy import constants as const


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
    return m_id[argsort].tolist(), list(zip(mm, ss))


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


def plot_grouped_cummulative_barchart(rankings, model_ids, out_dir):
    '''
    Plots a grouped barchart of the rankings

    Parameters
    ----------
    rankings : ndarray
        An array of the rankings [num_models x num_cases].
    model_ids : list
        List of model ids.
    out_dir : str
        Path to save the plot.

    Returns
    -------
    None.

    '''
    n, m = rankings.shape
    counts = np.zeros((n, n))
    for i, rank in enumerate(rankings):
        for j, model in enumerate(model_ids):
            counts[i, j] = np.count_nonzero(rankings[i, :] == (j + 1))
    counts = counts / m
    for i in range(n):
        for j in range(1, n):
            counts[i, j] = counts[i, j - 1] + counts[i, j]
    fig, ax = plt.subplots()
    x = np.arange(len(model_ids), dtype=np.int16)
    width = 0.15
    pos = x - 3 * width / 2
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

    fig_path = os.path.join(out_dir, 'ranking_cummulative_bar_chart.svg')
    plt.savefig(fig_path, format='svg')
    plt.close(fig)


def plot_grouped_barchart(rankings, model_ids, out_dir):
    '''
    Plots a grouped barchart of the rankings

    Parameters
    ----------
    rankings : ndarray
        An array of the rankings [num_models x num_cases].
    model_ids : list
        List of model ids.
    out_dir : str
        Path to save the plot.

    Returns
    -------
    None.

    '''
    n, m = rankings.shape
    counts = np.zeros((n, n))
    for i, rank in enumerate(rankings):
        for j, model in enumerate(model_ids):
            counts[i, j] = np.count_nonzero(rankings[i, :] == (j + 1))
    counts = counts / m
    fig, ax = plt.subplots()
    x = np.arange(len(model_ids), dtype=np.int16)
    width = 0.15
    pos = x - 3 * width / 2
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


def plot(y, mark, lab, dy=None):
    '''
    Plots y-values.

    Parameters
    ----------
    y : ndarray
        y-values.
    mark : char
        Marker symbol.
    lab : str
        Plot series label.
    dy : float
        y-values error.

    Returns
    -------
    None.

    '''
    if dy is not None:
        marker = '.'
        x = np.indices(y.shape)[0]
        plt.fill_between(x, y - dy, y + dy, color='gray', alpha=0.4)
    else:
        marker = mark
        plt.plot(y, marker=marker, label=lab, markersize=2, linestyle='')
    plt.ylim(-30, 0)
    # plt.xlim(300, 450)


def plot_matrix(data, name, labels, out_dir, symbols='ox+|_', scale='linear'):
    '''
    Plots a 2d array as separate plot series.

    Parameters
    ----------
    data : ndarray
        The 2d array [num_series x num_values].
    name : str
        Y-axis label.
    labels : list(str)
        Labels for each series.
    out_dir : str
        Path to save the plot.
    symbols : str, optional
        String of symbols to use as markers. The default is 'ox+|_'.
    scale : str, optional
        Y-axis scale. The default is 'linear'.

    Returns
    -------
    None.

    '''
    fig = plt.figure()
    for i, y in enumerate(data):
        plt.plot(y, marker=symbols[i], linestyle='', label=labels[i])
        plt.xlabel('case')
        plt.ylabel(name)
        plt.yscale(scale)
        plt.legend()
    fig_path = os.path.join(out_dir, '{}.svg'.format(name))
    plt.savefig(fig_path, format='svg')
    plt.close(fig)


def plot_cases(y, err_pred, prediction, labels,
               out_dir, cases, symbols='ox+|_'):
    '''
    Plots specific cases. Actual spectrum and prediction

    Parameters
    ----------
    y : ndarray
        Actual spectrum.
    err_pred : ndarray
        Error predictions.
    prediction : ndarray
        Prediction spectra [num_models x num_cases].
    labels : list(str)
        Labels for each series.
    out_dir : str
        Path to save the plot.
    cases : list(int)
        List of indices of cases to plot.
    symbols : str, optional
        String of symbols to use as markers. The default is 'ox+|_'.

    Returns
    -------
    None.

    '''
    c = const.c.cgs.value
    me = const.m_e.cgs.value
    h = const.h.cgs.value
    x_grid = np.linspace(-15, 10, 500)
    x_norm = np.log10(me * c**2 / h)
    xp = x_grid + x_norm

    y_pred = common.de_normalize(prediction)
    y_d = common.de_normalize(y)    # index 1 means the spectrum values
    # print(len(xp),len(y_d[0]))
    for i in cases:
        y_i = y_d[i]
        figsize = (8,4)
        fig = plt.figure(i,figsize=figsize)
        # plot(y_i, symbols[0], 'Atheνa')
        # plt.plot(xp, y_i[:-1], '-', label='ATHEνA')
        plt.plot(xp, y_i, 'k-', label='ATHEνA')
        plt.ylabel('log(vFv) [arbitery units]')
        plt.xlabel('log(v) [Hz]')
        for j, y_p in enumerate(y_pred):
            if err_pred is None:
                # plot(y_p[i], symbols[j + 1], labels[j])
                # plt.plot(xp,y_p[i, :-1],  marker=symbols[j + 1],label=labels[j],markersize=2, linestyle='')
                plt.plot(xp,y_p[i],color='orange',  marker=symbols[j + 1],label=labels[j],markersize=2, linestyle='',alpha=1.0)
            else:
                plot(y_p[i, :-1], symbols[j + 1], labels[j], err_pred[i, j])
        plt.ylim([-5.5,-3.0])
        plt.legend()
        fig_path = os.path.join(out_dir, '{}.svg'.format(i))
        plt.savefig(fig_path, format='svg')
        plt.close(fig)


def plot_all_cases(y, err_pred, prediction, labels, out_dir, symbols='ox+|_'):
    '''
    Plots all cases. Actual spectrum and prediction

    Parameters
    ----------
    y : ndarray
        Actual spectrum.
    err_pred : ndarray
        Error predictions.
    prediction : ndarray
        Prediction spectra [num_models x num_cases].
    labels : list(str)
        Labels for each series.
    out_dir : str
        Path to save the plot.
    symbols : str, optional
        String of symbols to use as markers. The default is 'ox+|_'.

    Returns
    -------
    None.

    '''
    y_pred = common.de_normalize(prediction)
    y_d = common.de_normalize(y)    # index 1 means the spectrum values
    for i, y_i in enumerate(y_d):
        fig = plt.figure(i)
        plot(y_i, symbols[0], 'actual')
        plt.ylabel('log(vFv)')
        plt.xlabel('log(v) [Hz]')
        plt.ylim([-15,-5])
        # plt.xlim([10,30])
        for j, y_p in enumerate(y_pred):
            if err_pred is None:
                plot(y_p[i], symbols[j + 1], labels[j])
            else:
                plot(y_p[i], symbols[j + 1], labels[j], err_pred[i, j])
        plt.legend()
        fig_path = os.path.join(out_dir, '{}.svg'.format(i))
        plt.savefig(fig_path, format='svg')
        plt.close(fig)


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
        plot_options = config['plot_options']
        paths = config['paths']

        working_dir = os.path.abspath(paths['working_dir'])
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

        dataset_path = dataset['path']
        n_features = dataset['inputs']
        # test_set is empty because ratio is 0
        input_set, _ = common.load_data(dataset_path, n_features, 0)

        input_set[1] = common.normalize_clip(input_set[1])
        # input_set[1] = common.normalize_clip2(input_set[1])

        models = []
        model_names = []
        for file in paths['models']:
            model = tf.keras.models.load_model(file)
            models.append(model)
            model_names.append(model.name)

        if plot_options.get('error', False):
            error_model = tf.keras.models.load_model(config['error_model'])
        else:
            error_model = None

        plot_rankings = plot_options.get('rankings', True)
        plot_all = plot_options.get('all', False)
        cases = plot_options.get('cases', [])

        prediction = common.calculate_predictions(models, input_set)

        metrics = {"dtw": dtw.distance,
                   "integral": common.integral_error,
                   "kolmogorov_smirnov": common.kolmogorov_smirnov_error,
                   "mse": metrics.mean_squared_error}

        choice = metrics[config.get('metric', 'kolmogorov_smirnov')]

        error_metric = common.calculate_error(input_set[1], prediction, choice)

        rank_of_mean, stats = mean_error_ranking(error_metric, model_names)

        mean_rank, mean_val, rankings = mean_ranking(error_metric, model_names)

        # np.savetxt(os.path.join(working_dir, 'mean_rank.txt'), mean_rank)
        # np.savetxt(os.path.join(working_dir, 'mean_val.txt'), mean_val)

        plot_matrix(
            error_metric,
            'error',
            model_names,
            working_dir,
            scale='log')

        if plot_all == True:
            # if error_model is not None:
            #     err_pred = error_model.predict(input_set[0])
            # else:
            #     err_pred = None
            plot_all_cases(
                input_set[1],
                None,
                prediction,
                model_names,
                working_dir)
        elif cases:
            # if error_model is not None:
            #     err_pred = error_model.predict(input_set[0])
            # else:
            #     err_pred = None
            plot_cases(
                input_set[1],
                None,
                prediction,
                model_names,
                working_dir,
                cases)

        if plot_rankings == True:
            plot_grouped_barchart(rankings, model_names, working_dir)
            plot_grouped_cummulative_barchart(
                rankings, model_names, working_dir)
            # plot_matrix(rankings, 'ranking', model_names, working_dir)
