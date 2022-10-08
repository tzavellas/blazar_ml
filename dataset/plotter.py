import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


class Plotter:
    _LWIDTH                 = 1
    _YLIM                   = (-16, 0)
    _MIN_LOG                = 1e-16
    _FORT81                 = 'fort.81'
    _FORT81_MAIN_CSV        = 'fort.81.main.csv'
    _FORT81_BASELINE_CSV    = 'fort.81.baseline.csv'
    _CSV_LABELS             = ('x', 'x^2*n(x)')
    _LATEX_LABELS           = (r'$x$', r'$x^2 \cdot n(x)$')


    def __init__(self):
        pass

    @staticmethod
    def diff(actual, baseline):
        '''
        Subtracts a baseline spectrum from an actual spectrum.
            Parameters:
                actual (tuple) : The actual spectrum
                baseline (tuple) : The baseline spectrum
        '''
        logger = logging.getLogger(__name__)

        logger.debug('Diff spectrum')
        x = np.array(actual[0])
        y1 = np.power(10, np.array(actual[1]))
        y2 = np.power(10, np.array(baseline[1]))

        dy = y1 - y2
        tmp = np.where(dy > Plotter._MIN_LOG, dy, Plotter._MIN_LOG)
        y = np.log10(tmp, where=tmp > 0)
        return x, y

    @staticmethod
    def plot_spectrum(x, y, id, file, labels=_LATEX_LABELS, lwidth=_LWIDTH):
        '''
        Plots x and y. If a file path is passed it saves the plot to the file path.
            Parameters:
                x (list):               x-axis values.
                y (list):               y-axis values.
                id (int):               Used to label the plot as "run <id>".
                file (str):             File path to save the plot. Default None -> does not save the figure.
                labels(tuple):          The labels of the x and y axis. Default x -> "x", y -> "x^2 * n(x)".
                lwidth(int):            Plot line width.
        '''
        logger = logging.getLogger(__name__)
        logger.info('Saving plot {}'.format(file))
        fig = plt.figure(str(id))
        plt.plot(x, y, ls='-', lw=lwidth, label='{}'.format(id))
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.ylim(*Plotter._YLIM)
        plt.title(r'run {}'.format(id))
        plt.savefig(file)
        plt.close(fig)

    @staticmethod
    def append_spectrum(df_main, df_base, id, lwidth=_LWIDTH):
        '''
        Wrapper function. Given a fort.81 file, extracts the steady state spectrum, saves it in a CSV file and plots it.
            Parameters:
                df_main (pandas.Dataframe):     Dataframe of the main spectrum.
                df_base (pandas.Dataframe):     Dataframe of the baseline spectrum.
                id (int):                       Used to label the plot in the legend.
                lwidth(int):                    Plot line width.
        '''
        x1 = df_main[Plotter._CSV_LABELS[0]]
        y1 = df_main[Plotter._CSV_LABELS[1]]
        y2 = df_base[Plotter._CSV_LABELS[1]]
        x, y = Plotter.diff((x1, y1), (x1,y2))
        plt.plot(x, y, ls='-', lw=lwidth, label='{}'.format(id))
        plt.ylim(*Plotter._YLIM)

    @staticmethod
    def aggregate_plots(output, working_dir, legend=False):
        '''
        Crawls working directory, reads each steady state and appends it in a single plot file.
            Parameters:
                output (str):           The aggregate plot file.
                working_dir (str):      The working directory.
                legend (bool):          If true, shows a legend in the plot file.
        '''
        logger = logging.getLogger(__name__)
        err = 0

        output = os.path.join(working_dir, output)
        if os.path.exists(output):
            logger.warning('Plot {} exists. Removing...'.format(output))
            os.remove(output)

        for (root, dirs, files) in os.walk(working_dir, topdown=False):
            main = Plotter._FORT81_MAIN_CSV
            baseline = Plotter._FORT81_BASELINE_CSV
            if (main in files) and (baseline in files):
                run_id = int(os.path.basename(root))
                main = os.path.join(root, main)
                baseline = os.path.join(root, baseline)

                try:
                    logger.debug('Reading {} and {} ...'.format(main, baseline))
                    df_main = pd.read_csv(main)
                    df_base = pd.read_csv(baseline)
                    logger.debug('Appending spectrum')
                    Plotter.append_spectrum(df_main, df_base, run_id)
                except BaseException as e:
                    logger.error('Reading {} and {}: {}'.format(main, baseline, e))
                    err = err + 1

        
        if legend:
            plt.legend()

        logger.info('Saving figure {} ...'.format(output))
        plt.savefig(output)

        return err
