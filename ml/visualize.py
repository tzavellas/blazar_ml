import argparse
import common
from dtaidistance import dtw
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
from astropy import constants as const
from ipywidgets import interact, widgets


def getColor(c, N, idx):
    cmap = mpl.colormaps[c]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    return cmap(norm(idx))


class SpectraPlotter:
    def __init__(self, y, y_pred, label, model_labels, figsize=(
            8, 4), x_grid=np.linspace(-15, 10, 500), ylim=[-5.5, -3.0]):
        self.y = y
        self.y_pred = y_pred
        self.label = label
        self.m_labels = model_labels

        self.figsize = figsize
        x_norm = np.log10(const.m_e.cgs.value *
                          const.c.cgs.value**2 / const.h.cgs.value)
        self.x_values = x_grid + x_norm
        self.ylim = ylim
        self.ax = None

        self.plt_colors = [getColor("Spectral", 20, i) for i in range(20)]
        self.colors = [
            self.plt_colors[0],
            self.plt_colors[4],
            self.plt_colors[18]]
        self.marks = 'ox+|_'

    def plot_dataset_spectrum(self, case_index):
        _, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.plot(
            self.x_values,
            self.y[case_index],
            color='black',
            label=self.label,
            linestyle='None',
            marker='|',
            markersize=3)
        self.ax.set_ylabel('log(vFv) [arbitery units]')
        self.ax.set_xlabel('log(v) [Hz]')
        ymax = np.max(self.y[case_index])
        self.ax.set_ylim([ymax-6,ymax+0.5])

    def plot_prediction_spectrum(self, case_index, model_index):
        self.ax.plot(
            self.x_values,
            self.y_pred[model_index][case_index],
            color=self.colors[model_index],
            label=self.m_labels[model_index],
            linestyle='None',
            marker=self.marks[model_index],
            markersize=3)

    def plot_spectra(self, case_index, model_index):
        self.plot_dataset_spectrum(case_index)
        self.plot_prediction_spectrum(case_index, model_index)
        self.ax.legend()

    def save(self, fig_path):
        self.ax.legend()
        plt.savefig(fig_path)
        plt.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description='Loads a dataset and plots actual and predicted curves of each test case.')
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
        plot_options = config['plot_options']
        paths = config['paths']
        dataset_path = dataset['path']
        n_features = dataset['inputs']

        # test_set is _ because ratio is 0.0
        input_set, _ = common.load_data(dataset_path, n_features, 0.0)
        y = input_set[1]  # Keep a copy of the denormalized spectrum values
        input_set[1] = common.normalize_clip(input_set[1])

        # Load models from files
        models = []
        model_names = []
        for file in paths['models']:
            model = tf.keras.models.load_model(file)
            models.append(model)
            model_names.append(model.name)
        model_labels = [model.name for m, model in enumerate(models)]

        # Make predictions and denormalize the output
        prediction = common.calculate_predictions(models, input_set)
        y_pred = common.de_normalize(prediction)

        if plot_options.get('all', True):
            n, _ = input_set[1].shape
            cases = [i for i in range(n)]
        elif plot_options.get('cases', [0]):
            cases = plot_options['cases']

        # Create a plotter object
        plotter = SpectraPlotter(y, y_pred, 'AthevA', model_labels)

        # Jupyter interactive section. Has no effect without Jupyter
        if plot_options.get('interactive', False):
            case_slider = widgets.SelectionSlider(
                options=cases,
                value=cases[0],
                description='Case Index:',
                continuous_update=False)

            choices = [(model.name, m) for m, model in enumerate(models)]
            dropdown = widgets.Dropdown(
                options=choices, value=0, description='Model:')

            interact(
                plotter.plot_spectra,
                case_index=case_slider,
                model_index=dropdown)
        else:  # Plots all cases defined in the config file
            working_dir = os.path.abspath(paths['working_dir'])
            os.makedirs(working_dir, exist_ok=True)
            for case in cases:
                plotter.plot_dataset_spectrum(case)
                for m in range(len(models)):
                    plotter.plot_prediction_spectrum(case, m)
                plotter.save(os.path.join(working_dir, f'{case}.svg'))
    return 0


if __name__ == "__main__":
    try:
        args = parse_arguments(sys.argv[1:])
        sys.exit(main(args))
    except argparse.ArgumentError:
        sys.exit(1)
