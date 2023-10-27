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
from scipy.optimize import linear_sum_assignment


class ModelErrorAnalyzer:
    def __init__(self, error, model_ids):
        self.error = error
        self.m_id = np.array(model_ids)

    @staticmethod
    def is_double_stochastic(matrix):
        """Checks if a matrix is double stochastic

        Args:
            matrix (nd.array): The matrix

        Returns:
            bool: True if it is double stochastic
        """
        return np.allclose(
            matrix.sum(
                axis=0),
            1) and np.allclose(
            matrix.sum(
                axis=1),
            1)

    @staticmethod
    def average_ranking(error, model_ids):
        """ Calculates the mean value and std of the error per model.
        Returns a ranking of the mean value, a list of ids sorted by the ranking and
        a list of statistics (mean, std) sorted by the ranking.

        Args:
            error (nd.array): An [num_models x num_cases] matrix with the error metric.
            model_ids (list): List of model ids.

        Returns:
            tuple(list, list, list): The ranking, the ranked ids, and the ranked statistics.
        """
        mm = np.mean(error, axis=1)
        ss = np.std(error, axis=1)
        argsort = np.argsort(mm)
        return argsort, model_ids[argsort].tolist(), list(
            zip(mm[argsort], ss[argsort]))

    @staticmethod
    def birkhoff_von_neumann_decomposition(matrix):
        """Performs Birkhoff-Von Neumann decomposition

        Args:
            matrix (nd.array): A double stochastic matrix

        Returns:
            tuple(list, list): A list of permutation matrices and their corresponding weights.
        """
        assert ModelErrorAnalyzer.is_double_stochastic(
            matrix), "Matrix is not double stochastic"

        n = matrix.shape[0]
        permutation_matrices = []
        weights = []
        while not np.allclose(matrix, 0):
            r, c = linear_sum_assignment(-matrix)
            P = np.zeros_like(matrix)
            P[r, c] = 1

            weight = np.min(matrix[r, c])
            weights.append(weight)
            permutation_matrices.append(P)

            matrix = matrix - weight * P
        return permutation_matrices, weights

    @staticmethod
    def most_representative_ranking(error, model_ids):
        """Runs the Most Representative Ranking algorithm.
        Returns a ranking and a list of ids sorted by the ranking.

        Args:
            error (nd.array): An [num_models x num_cases] matrix with the error metric.
            model_ids (list): List of model ids.

        Returns:
            tuple(list, list, list): The ranking, the ranked ids and the counts matrix.
        """
        counts, m = ModelErrorAnalyzer.bin_counts(error)
        mat = counts / m
        permutation_matrices, weights = ModelErrorAnalyzer.birkhoff_von_neumann_decomposition(
            mat)
        max_weight_index = np.argmax(weights)
        choice = permutation_matrices[max_weight_index]

        ranking = np.argmax(choice, axis=1)
        return ranking, model_ids[ranking].tolist()

    @staticmethod
    def median_ranking(error, model_ids):
        """Calculates the median of the error per model. Returns a
        ranking of the median and a list of ids sorted by the ranking.

        Args:
            error (nd.array): An [num_models x num_cases] matrix with the error metric.
            model_ids (list): List of model ids.

        Returns:
            tuple(list, list): The ranking, the ranked ids
        """
        mm = np.median(error, axis=1)
        argsort = np.argsort(mm)
        return argsort, model_ids[argsort].tolist()

    @staticmethod
    def min_error_ranking(error, model_ids):
        mm = np.min(error, axis=1)
        argsort = np.argsort(mm)
        return argsort, model_ids[argsort].tolist()

    @staticmethod
    def max_error_ranking(error, model_ids):
        mm = np.max(error, axis=1)
        argsort = np.argsort(mm)
        return argsort, model_ids[argsort].tolist()

    @staticmethod
    def bin_counts(error):
        argsort = np.argsort(error.transpose())
        m = argsort.shape[0]
        n = argsort.shape[1]

        counts = np.zeros((n, n))
        for col in range(n):
            data = argsort[:, col]
            count = np.bincount(data, minlength=n)
            counts[:, col] = count
        return counts, m

    def average_algorithm(self):
        return self.average_ranking(self.error, self.m_id)

    def bv_algorithm(self):
        return self.most_representative_ranking(self.error, self.m_id)

    def counts(self):
        return self.bin_counts(self.error)[0]

    def median_algorithm(self):
        return self.median_ranking(self.error, self.m_id)

    def min_algorithm(self):
        return self.min_error_ranking(self.error, self.m_id)

    def max_algorithm(self):
        return self.max_error_ranking(self.error, self.m_id)


def plot_barchart(counts, title, plot_file=None):
    n = counts.shape[1]
    fig, ax = plt.subplots()
    bar_width = 0.25
    r = np.arange(counts.shape[0])
    for col in range(n):
        ax.bar(r + col * bar_width, counts[:, col],
               width=bar_width, label=model_names[col])
    ax.set_xticks(r + bar_width)
    ax.set_xticklabels([1, 2, 3])
    ax.set_title(title)
    ax.set_xlabel('Ranking')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    if plot_file:
        plt.savefig(plot_file)
        plt.close(fig)


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
        figsize = (8, 4)
        fig = plt.figure(i, figsize=figsize)
        # plot(y_i, symbols[0], 'Atheνa')
        # plt.plot(xp, y_i[:-1], '-', label='ATHEνA')
        plt.plot(xp, y_i, 'k-', label='ATHEνA')
        plt.ylabel('log(vFv) [arbitery units]')
        plt.xlabel('log(v) [Hz]')
        for j, y_p in enumerate(y_pred):
            if err_pred is None:
                # plot(y_p[i], symbols[j + 1], labels[j])
                # plt.plot(xp,y_p[i, :-1],  marker=symbols[j + 1],label=labels[j],markersize=2, linestyle='')
                plt.plot(xp,
                         y_p[i],
                         color='orange',
                         marker=symbols[j + 1],
                         label=labels[j],
                         markersize=2,
                         linestyle='',
                         alpha=1.0)
            else:
                plot(y_p[i, :-1], symbols[j + 1], labels[j], err_pred[i, j])
        plt.ylim([-5.5, -3.0])
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
        plt.ylim([-15, -5])
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

        # Check config dictionary for environment replacement
        try:
            config = common.check_environment(config)
        except ValueError as e:
            print(f'{e}')
            sys.exit(1)
        # Read config dictionaries
        dataset = config['dataset']
        plot_options = config['plot_options']
        paths = config['paths']

        label = config['label']
        working_dir = os.path.abspath(paths['working_dir'])
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

        dataset_path = dataset['path']
        n_features = dataset['inputs']
        # test_set is empty because ratio is 0
        input_set, _ = common.load_data(dataset_path, n_features, 0)

        input_set[1] = common.normalize_clip(input_set[1])
        # input_set[1] = common.normalize_clip2(input_set[1])

        metrics = {"dtw": dtw.distance,
                   "integral": common.integral_error,
                   "kolmogorov_smirnov": common.kolmogorov_smirnov_error,
                   "mse": metrics.mean_squared_error}

        choice = metrics[config.get('metric', 'kolmogorov_smirnov')]

        models = []
        model_names = []
        for file in paths['models']:
            model = tf.keras.models.load_model(file)
            models.append(model)
            model_names.append(model.name)

        prediction = common.calculate_predictions(models, input_set)

        error_metric = common.calculate_error(input_set[1], prediction, choice)

        analyzer = ModelErrorAnalyzer(error_metric, model_names)

        metric_ = config.get('metric', 'kolmogorov_smirnov')
        basename = f'{label}_{metric_}'
        report = os.path.join(working_dir, f'{basename}_report.txt')
        with open(report, 'w') as f:
            f.write(f'Metric: {metric_}\n\n')
            stats_ranking, stats_ids, stats = analyzer.average_algorithm()
            f.write(f'Average Ranking: {stats_ids} ({stats_ranking})\n')
            bv_ranking, bv_ids = analyzer.bv_algorithm()
            f.write(f'Birkhoff-Von Neumann Ranking: {bv_ids} ({bv_ranking})\n')
            med_ranking, med_ids = analyzer.median_algorithm()
            f.write(f'Median Ranking: {med_ids} ({med_ranking})\n')
            max_ranking, max_ids = analyzer.max_algorithm()
            f.write(f'Max Ranking: {max_ids} ({max_ranking})\n')
            min_ranking, min_ids = analyzer.min_algorithm()
            f.write(f'Min Ranking: {min_ids} ({min_ranking})\n')

        if plot_options['rankings']:
            plot_file = os.path.join(working_dir, f'{basename}_barchart.svg')
            counts = analyzer.counts()
            plot_barchart(counts, plot_options['title'], plot_file)
