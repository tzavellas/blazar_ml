#!/usr/bin/python

import argparse
from code_launcher import CodeLauncher
from datetime import datetime
from multiprocessing import Pool
import os
from pathlib import Path
from plotter import Plotter
import logging.config
import sys


_FILENAME = Path(__file__).stem


def main():
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
        return 1

    if not os.path.exists(args.executable):
        print('Executable {} does not exist'.format(args.executable))
        return 1

    if not os.path.exists(args.working_dir):
        print('Working directory {} does not exist'.format(args.working_dir))
        return 1

    if not os.path.exists(args.input):
        print('Input csv {} does not exist'.format(args.input))
        return 1

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
            return 1

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
            return 1
    logger.info('Done')
    return 0


if __name__ == "__main__":
    sys.exit(main())
