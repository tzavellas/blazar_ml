#!/usr/bin/env /home/mapet/Progs/anaconda3/envs/tf/bin/python

import argparse
import logging.config
import os
from pathlib import Path
import pandas as pd
import sys
from interpolator import Interpolator


_FILENAME = Path(__file__).stem


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Interpolates spectra and returns the interpolated values in a csv.')
    parser.add_argument('-o', '--output', default='interpolated.csv', type=str,
                        help='Interpolated values of all spectra. Default is "interpolated.csv".')
    parser.add_argument('-w', '--working-dir', default='output', type=str,
                        help='Root path where the individual spectra are stored. Default is "output".')
    parser.add_argument('-k', '--degree', default='1', type=int,
                        help='The degree of the spline fit. Default is 1.')
    parser.add_argument('-l', '--logging', type=str, default='logging.ini',
                        help='Log configuration. Default is logging.ini.')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as arg_e:
        print('parse_args: {}'.format(arg_e), file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.working_dir):
        print('Working directory {} does not exist'.format(args.working_dir))
        sys.exit(1)

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

    filename = args.output
    working_dir = os.path.abspath(args.working_dir)
    order = args.degree

    output = os.path.join(working_dir, filename)
    if os.path.exists(output):
        logger.warning('Plot {} exists. Removing...'.format(output))
        os.remove(output)

    err, out_dict = Interpolator.interpolate_spectra(working_dir, num=250, k=order)
    
    logger.debug('Storing dict in file {}...'.format(output))
    df = pd.DataFrame(out_dict)
    df.to_csv(output, index=False)
    sys.exit(err)
