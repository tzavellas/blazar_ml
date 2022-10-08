#!/usr/bin/env /home/mapet/Progs/anaconda3/envs/tf/bin/python

import argparse
from code_launcher import CodeLauncher
from datetime import datetime
from interpolator import Interpolator
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from pathlib import Path
from plotter import Plotter
import logging.config
import sys


_FILENAME = Path(__file__).stem


def normalize(data, min_val=-30, max_val=0):
    return (data - min_val) / (max_val - min_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generates the dataset given a sample of the input space.')
    parser.add_argument('-e', '--executable', type=str, required=True,
                        help='Full path to the program executable.')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='CSV file with the a sample of the input space.')
    parser.add_argument('-w', '--working-dir', type=str, default='output',
                        help='Root path where the dataset will be stored. Default is "output".')
    parser.add_argument('-f', '--format', type=str, default='png',
                        help='Spectrum image format. Default is png.')
    parser.add_argument('-l', '--logging', type=str, default='logging.ini',
                        help='Log configuration. Default is logging.ini.')
    parser.add_argument('-n', '--num-proc', type=int, default=None,
                        help='Number of processes to launch. Default is number of system threads')
    parser.add_argument('-p', '--plot-spectra', action='store_true', default=True,
                        help='Aggregate plot of all spectra in a file "spectra.png". Default is True.')
    parser.add_argument('-x', '--extra-args', default=[], nargs='*',
                        help='List of extra arguments to pass to the program. Default is [].')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as arg_e:
        print('parse_args: {}'.format(arg_e), file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.executable):
        print('Executable {} does not exist'.format(args.executable))
        sys.exit(1)

    if not os.path.exists(args.working_dir):
        print('Working directory {} does not exist'.format(args.working_dir))
        sys.exit(1)

    if not os.path.exists(args.input):
        print('Input csv {} does not exist'.format(args.input))
        sys.exit(1)

    logfile = os.path.join(args.working_dir, '{}.log'.format(_FILENAME))
    if os.path.exists(logfile):
        os.remove(logfile)

    try:
        logging_ini = args.logging
        print('Loading {} ...'.format(logging_ini))
        logging.config.fileConfig(logging_ini, defaults={
                                  'logfilename': '{}'.format(logfile)}, disable_existing_loggers=True)
        print('Logfile: {}'.format(logfile))
        logger = logging.getLogger(os.path.basename(__file__))
    except Exception as e:
        print('Failed to load config from {}. Exception {}'.format(logging_ini, e))
        logging.basicConfig(
            format='%(asctime)s %(name)s - %(levelname)s: %(message)s')
        logging.getLogger('matplotlib').disabled = True
        logging.getLogger('matplotlib.font_manager').disabled = True
        logger = logging.getLogger(os.path.basename(__file__))
        logger.setLevel(logging.DEBUG)

    launcher = CodeLauncher(exec_path=args.executable,
                            working_dir=args.working_dir,
                            input=args.input)

    rows = launcher.get_inputs_dataframe().shape[0]
    logger.info('Input csv contains {} rows'.format(rows))

    params = []
    for row in range(rows):
        params.append((row, args.extra_args, args.format))

    # start measuring time
    start_time = datetime.now()

    # run in parallel processes
    with Pool(processes=args.num_proc) as pool:
        try:
            ret = pool.starmap(launcher.run, iterable=params)
        except Exception as e:
            logger.error('starmap: {}'.format(e))
            sys.exit(1)

    # stop measuring time
    end_time = datetime.now()

    # logs total runtime duration
    logger.info('Total duration: {}'.format(end_time - start_time))

    inputs_dataframe = launcher.get_inputs_dataframe()
    for run_id, success, elapsed, elapsed_base in ret:
        inputs_dataframe.at[run_id, CodeLauncher.SUCCESS_KEY] = success
        inputs_dataframe.at[run_id, CodeLauncher.ELAPSED_TIME_KEY] = '{:.0f}'.format(
            elapsed)
        inputs_dataframe.at[run_id, CodeLauncher.ELAPSED_TIME_BASELINE_KEY] = '{:.0f}'.format(
            elapsed_base)
    # Write the dataframe to the output directory
    launcher.write_dataframe(inputs_dataframe, args.working_dir)

    if args.plot_spectra:
        spectra = 'spectra.{}'.format(args.format)
        err = Plotter.aggregate_plots(
            output=spectra, working_dir=args.working_dir, legend=True)
        if err:
            logger.error('Error plotting aggregate spectrum')
            sys.exit(1)

    interpolated_file = 'interpolated.csv'
    interpolated_file = os.path.join(args.working_dir, interpolated_file)
    if os.path.exists(interpolated_file):
        logger.warning('{} exists. Removing...'.format(interpolated_file))
        os.remove(interpolated_file)

    err, out_dict = Interpolator.interpolate_spectra(args.working_dir)

    logger.info('Storing dict in file {}...'.format(interpolated_file))
    interpolated = pd.DataFrame(out_dict)
    interpolated.to_csv(interpolated_file, index=False)

    normalized = pd.DataFrame()
    skipped = []

    rows, cols = interpolated.shape
    for i in range(1, cols):
        run = inputs_dataframe.loc[i-1, 'run']
        x = inputs_dataframe.loc[i-1, 'radius':'slelints']
        s = 'y_{}'.format(int(run))
        if (s in interpolated.columns):
            y = interpolated.loc[:, s]
            y_n = normalize(y)
            w = y_n.append(pd.Series([np.amax(y_n)]), ignore_index=True)
            z = x.append(w, ignore_index=True)
            normalized = normalized.append(z, ignore_index=True)
        else:
            skipped.append(s)
    logger.info('Skipped: {}'.format(skipped))

    normalized_file = 'normalized.csv'
    normalized_file = os.path.join(args.working_dir, normalized_file)
    if os.path.exists(normalized_file):
        logger.warning('{} exists. Removing...'.format(normalized_file))
        os.remove(normalized_file)

    normalized.to_csv(normalized_file, header=False, index=False)

    logger.info('Done')
    sys.exit(0)