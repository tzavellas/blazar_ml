#!/usr/bin/python

import argparse
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

plot_fname              = 'plot'
program_output_fname    = 'fort.81'
steady_state_fname      = 'steady_state.csv'


def extract_steady_state(file):
    '''
    Reads an output file and extracts the steady state values.
        Parameters:
            file (str) : The fort.81 file
    '''
    x = []
    y = []
    with open(file) as f:
        for line in f:                      # read line by line
            tokens = line.rstrip().split()  # tokenize line
            if len(tokens) == 4:            # new block starts
                x.clear()
                y.clear()
                continue
            else:                           # same block
                x.append(float(tokens[0]))
                y.append(float(tokens[1]))
    return x, y


def save_spectrum(x, y, file, labels=('x', 'x^2*n(x)')):
    '''
    Saves x and y in a csv file.
        Parameters:
            x (list):               x-axis values
            y (list):               y-axis values
            file (str):             File path to save the data.
            labels (tuple):         The labels of the csv columns. Default x -> "x",  y -> "x^2*n(x)"
    '''
    df = pd.DataFrame()
    df[labels[0]] = x
    df[labels[1]] = y
    df.to_csv(file, index=True, index_label='i',
              float_format='%.6e', line_terminator='\n')
    return


def plot_spectrum(x, y, id, labels=(r'$x$', r'$x^2 \cdot n(x)$'), marker=3, lwidth=1, file=None):
    '''
    Plots x and y. If a file path is passed it saves the plot to the file path.
        Parameters:
            x (list):               x-axis values.
            y (list):               y-axis values.
            id (int):               Used to label the plot as "run <id>".
            labels(tuple):          The labels of the x and y axis. Default x -> "x", y -> "x^2 * n(x)".
            marker(int):            Marker size. Default 3.
            file (str):             File path to save the plot. Default None -> does not save the figure.
    '''
    plt.plot(x, y, ls='-', lw=lwidth, label='{}'.format(id))
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.ylim(-16,0)
    plt.title(r'run {}'.format(id))
    if file is not None:
        plt.savefig(file)
    return


def append_spectrum(df, id, marker=3, lwidth=1):
    '''
    Wrapper function. Given a fort.81 file, extracts the steady state spectrum, saves it in a CSV file and plots it.
        Parameters:
            df (pandas.Dataframe):  Dataframe of the current .
            id (int):               Used to label the plot in the legend.
            marker(int):            Marker size. Default 3.
    '''
    x = df['x']
    y = df['x^2*n(x)']
    plt.plot(x, y, ls='-', lw=lwidth, label='{}'.format(id))
    plt.ylim(-16,0)
    return


def save(id, working_dir, img_format):
    '''
    Wrapper function. Given a fort.81 file, extracts the steady state spectrum, saves it in a CSV file and plots it.
        Parameters:
            id (int):               run id of the execution.
            working_dir (str):      The working directory.
            img_format (str):       Image format of the spectrum plot.
    '''
    fort81  = '{}/{}'.format(working_dir, program_output_fname)
    img     = '{}/{}.{}'.format(working_dir, plot_fname, img_format)
    ss      = '{}/{}'.format(working_dir, steady_state_fname)

    x, y = extract_steady_state(fort81)
    save_spectrum(x, y, file=ss)
    plot_spectrum(x, y, id, file=img)
    return


def aggregate_plots(output, working_dir, img_format, legend, logger):
    '''
    Crawls working directory, reads each steady state and appends it in a single plot file.
        Parameters:
            output (str):           The plot file.
            working_dir (str):      The working directory.
            img_format (str):       Image format of the spectrum plot.
            legend (bool):          If true, shows a legend in the plot file.
    '''
    err = False
    dir = os.fsencode(working_dir)

    output = os.path.abspath('{}/{}.{}'.format(working_dir, output, img_format))
    if os.path.exists(output):
        logger.debug('{} exists. Removing...'.format(output))
        os.remove(output)

    for file in os.listdir(dir):
        run_id = os.fsdecode(file)
        scenario = os.path.abspath('{}/{}/steady_state.csv'.format(working_dir, run_id))

        if not os.path.exists(scenario):
            logger.warning('Scenario {} does not exist'.format(scenario))
            logger.warning('Skipping Run {} from plot'.format(run_id))
            continue

        logger.debug('Reading {} ...'.format(scenario))

        try:
            df = pd.read_csv(scenario)
        except BaseException as e:
            logger.error('read_csv: {}'.format(e))
            err = True

        logger.debug('Appending spectrum from Run {} ...'.format(run_id))
        append_spectrum(df, run_id)

    if legend:
        plt.legend()

    logger.debug('Saving figure {} ...'.format(output))
    plt.savefig(output, format=img_format)
    return err


def init_logger(logfile, log_level=logging.DEBUG):
    '''
    Initializes the logger.
        Parameters:
            logfile (str):                  Path to the output logfile.
            log_level (enum):               Logger log level. Default is DEBUG.
    '''
    # disables matplotlib logging to avoid spam
    logging.getLogger('matplotlib').disabled = True
    logging.getLogger('matplotlib.font_manager').disabled = True

    logger = logging.getLogger()

    # Set log format time - log level : msg
    formatter= logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    
    # Attach a file logger
    file_handler = logging.FileHandler(filename=logfile, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Attach a console logger
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Set log level
    logger.setLevel(log_level)

    return logger


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Plots spectra in a single file.')
    parser.add_argument('-o', '--output', default='spectra', type=str, 
                        help='Aggregate plot of all spectra. Default is "spectra.png".')
    parser.add_argument('-w', '--working-dir', default='output', type=str,
                        help='Root path where the individual spectra are stored. Default is "output".')
    parser.add_argument('-f', '--format', type=str, default='png',
                        help='Spectrum image format. Default is png.')
    parser.add_argument('--legend', action='store_true', default=False,
                        help='Adds legend to the spectra plot. Default is false.')
    parser.add_argument('-l', '--logging', type=str, default='generate_dataset.log',
                        help='Log file. Default is generate_dataset.log.')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as arg_e:
        print('parse_args: {}'.format(arg_e), file=sys.stderr)
        sys.exit(1)
    
    logfile = os.path.abspath(args.logging)
    if os.path.exists(logfile):
        os.remove(logfile)
    print('Logfile: {}'.format(logfile))

    logger = init_logger(logfile)

    ret = aggregate_plots(args.output, args.working_dir, args.format, args.legend, logger)
    
    sys.exit(ret)
