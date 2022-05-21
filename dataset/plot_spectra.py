#!/usr/bin/python

import argparse
import logging.config
import os
from pathlib import Path
from plotter import Plotter
import sys

_FILENAME = Path(__file__).stem

def main():
    parser = argparse.ArgumentParser(
        description='Plots spectra in a single file.')
    parser.add_argument('-o', '--output', default='spectra', type=str, 
                        help='Aggregate plot file of all spectra. Default is "spectra.png".')
    parser.add_argument('-w', '--working-dir', default='output', type=str,
                        help='Root path where the individual spectra are stored. Default is "output".')
    parser.add_argument('-l', '--logging', type=str, default='logging.ini',
                        help='Log configuration. Default is logging.ini.')
    parser.add_argument('--legend', action='store_true', default=False,
                        help='Adds legend to the spectra plot. Default is false.')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as arg_e:
        print('parse_args: {}'.format(arg_e), file=sys.stderr)
        return 1

    if not os.path.exists(args.working_dir):
        print('Working directory {} does not exist'.format(args.working_dir))
        return 1

    logfile = os.path.join(args.working_dir, '{}.log'.format(_FILENAME))
    if os.path.exists(logfile):
        os.remove(logfile)

    try:
        logging_ini = args.logging
        print('Loading {} ...'.format(logging_ini))
        logging.config.fileConfig(logging_ini, defaults={'logfilename': '{}'.format(logfile)})
        print('Logfile: {}'.format(logfile))
        logger = logging.getLogger(os.path.basename(__file__))
    except Exception as e:
        print('Failed to load config from {}. Exception {}'.format(logging_ini, e))
        logging.basicConfig(format='%(asctime)s %(name)s - %(levelname)s: %(message)s')
        logger = logging.getLogger(os.path.basename(__file__))
        logger.setLevel(logging.DEBUG)
    

    filename, file_extension = os.path.splitext(args.output)
    if not file_extension:
        logger.info('Output file without extension. Assuming .png')
        filename = '{}.png'.format(filename)
        
    working_dir = os.path.abspath(args.working_dir)
    ret = Plotter.aggregate_plots(filename, working_dir, legend=args.legend)

    return ret


if __name__ == "__main__":
    sys.exit(main())
