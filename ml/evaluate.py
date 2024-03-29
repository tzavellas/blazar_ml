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
from scipy.stats import wasserstein_distance


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
            mat.transpose())
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
        return argsort, model_ids[argsort].tolist(), mm[argsort]

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

    def frequencies(self):
        counts_, total = self.bin_counts(self.error)
        return counts_ / total

    def median_algorithm(self):
        return self.median_ranking(self.error, self.m_id)

    def min_algorithm(self):
        return self.min_error_ranking(self.error, self.m_id)

    def max_algorithm(self):
        return self.max_error_ranking(self.error, self.m_id)

    def to_file(self, file, used_metric):
        with open(file, 'w') as f:
            f.write(f'Metric: {used_metric}\n\n')
            bv_ranking, bv_ids = self.bv_algorithm()
            f.write(f'Birkhoff-Von Neumann Ranking: {bv_ids} ({bv_ranking})\n')
            stats_ranking, stats_ids, stats = self.average_algorithm()
            f.write(f'Average Ranking: {stats_ids} ({stats_ranking})\n')
            med_ranking, med_ids, median_values = self.median_algorithm()
            f.write(f'Median Ranking: {med_ids} ({med_ranking})\n')

            f.write(f'Median error: {med_ids} -> {median_values}\n')


def plot_barchart(counts, labels, plot_file=None, title=None):
    n = counts.shape[1]
    fig, ax = plt.subplots()
    bar_width = 0.25
    r = np.arange(counts.shape[0])
    for col in range(n):
        ax.bar(r + col * bar_width, counts[col, :],
               width=bar_width, label=labels[col])
    ax.set_xticks(r + bar_width)
    ax.set_xticklabels([1, 2, 3])
    if title:
        ax.set_title(title)
    ax.set_xlabel('Ranking')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    if plot_file:
        plt.savefig(plot_file)
        plt.close(fig)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description='Loads a dataset and several models and generates an evaluation report.')
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
            sys.exit(1)
        # Read config dictionaries
        dataset = config['dataset']
        plot_options = config['plot_options']
        paths = config['paths']

        label = config['label']
        working_dir = os.path.join(os.path.abspath(paths['working_dir']), label)
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

        dataset_path = dataset['path']
        n_features = dataset['inputs']
        # test_set is _ because ratio is 0
        input_set, _ = common.load_data(dataset_path, n_features, 0.0)

        models = []
        model_names = []
        for file in paths['models']:
            model = tf.keras.models.load_model(file)
            models.append(model)
            model_names.append(model.name)

        # Make predictions and denormalize the output
        prediction = common.calculate_predictions(models, input_set)
        y_pred = common.de_normalize(prediction)
        y = input_set[1]

        if not plot_options['all']:
            if plot_options.get('cases', None):
                indices = plot_options['cases']
                y_pred = y_pred[:, indices, :]
                y = y[indices, :]

        available_metrics = {
                "dtw": dtw.distance_fast,
                "kolmogorov_smirnov": common.kolmogorov_smirnov_error,
                "wasserstein": wasserstein_distance,
                "mse": metrics.mean_squared_error,
                "mae": metrics.mean_absolute_error}

        metric_ = config.get('metric', 'all')
        if metric_ == 'all':
            for metric_, choice in available_metrics.items():
                error_metric = common.calculate_error(y, y_pred, choice)

                analyzer = ModelErrorAnalyzer(error_metric, model_names)

                basename = f'{label}_{metric_}'
                report = os.path.join(working_dir, f'{basename}_report.txt')
                plot_file = os.path.join(working_dir, f'{basename}_barchart.svg')

                analyzer.to_file(report, metric_)
                frequencies = analyzer.frequencies() * 100
                with open(report, 'a') as f:
                    f.write(f'\nRanking percentages:\n')
                    for i, name_ in enumerate(model_names):
                        freq_string = [f'{value:.2f} ' for value in frequencies[i]]
                        f.write(f'{name_:<4} -> {freq_string}\n')
                plot_barchart(frequencies, model_names, plot_file)
        else:
            choice = available_metrics[metric_]
            error_metric = common.calculate_error(y, y_pred, choice)

            analyzer = ModelErrorAnalyzer(error_metric, model_names)

            basename = f'{label}_{metric_}'
            report = os.path.join(working_dir, f'{basename}_report.txt')
            plot_file = os.path.join(working_dir, f'{basename}_barchart.svg')

            analyzer.to_file(report, metric_)
            frequencies = analyzer.frequencies() * 100
            with open(report, 'a') as f:
                f.write(f'\nRanking percentages:\n')
                for i, name_ in enumerate(model_names):
                    freq_string = [f'{value:.2f} ' for value in frequencies[i]]
                    f.write(f'{name_:<4} -> {freq_string}\n')
            plot_barchart(frequencies, model_names, plot_file)

        r_squared = np.mean(common.calculate_error(y, y_pred, common.r_squared), axis=1)
        r2_file = os.path.join(working_dir, f'r2_report.txt')
        with open(r2_file, 'w') as f:
            f.write(f'Mean R2: {model_names} -> {r_squared}')

    return 0


if __name__ == "__main__":
    try:
        args = parse_arguments(sys.argv[1:])
        sys.exit(main(args))
    except argparse.ArgumentError:
        sys.exit(1)
