import argparse
import json
import logging.config
import os
import pandas as pd
import sys
from interpolator import Interpolator


if __name__ == "__main__":
    filename = os.path.basename(__file__).split('.')[0]

    parser = argparse.ArgumentParser(
        description='Interpolates spectra and returns the interpolated values in a csv.')
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        required=True,
        help='Config file.')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as arg_e:
        print(arg_e)
        sys.exit(1)

    with open(args.config) as config:
        config = json.loads(config.read())

        dataset = config['dataset']
        interpolation = config['interpolation']
        clamp = config['clamp']

        working_dir = os.path.abspath(dataset['working_dir'])
        if not os.path.exists(working_dir):
            print('Working directory {} does not exist'.format(working_dir))
            sys.exit(1)

        logfile = os.path.join(working_dir, '{}.log'.format(filename))
        print('Logfile: {}'.format(logfile))
        if os.path.exists(logfile):
            os.remove(logfile)

        try:
            logging_ini = dataset['logging']
            print('Loading {} ...'.format(logging_ini))
            logging.config.fileConfig(
                logging_ini, defaults={
                    'logfilename': '{}'.format(logfile)})
            logger = logging.getLogger(os.path.basename(__file__))
        except Exception as e:
            print(
                'Failed to load config from {}. Exception {}'.format(
                    logging_ini, e))
            logging.basicConfig(
                format='%(asctime)s %(name)s - %(levelname)s: %(message)s')
            logger = logging.getLogger(os.path.basename(__file__))
            logger.setLevel(logging.DEBUG)

        output = os.path.join(working_dir, dataset['output'])

        degree = interpolation['degree']
        x_start = interpolation['x_start']
        x_end = interpolation['x_end']
        n_points = interpolation['n_points']
        clamped = interpolation['clamped']

        err, out_dict = Interpolator.interpolate_spectra(working_dir,
                                                         x_start,
                                                         x_end,
                                                         n_points,
                                                         degree,
                                                         clamp if clamped else None)
        logger.debug('Storing dict in file {}...'.format(output))
        df = pd.DataFrame(out_dict)
        df.to_csv(output, index=False, na_rep='NaN')
        sys.exit(err)
