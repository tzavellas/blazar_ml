import logging
import numpy as np
import os
import pandas as pd
from scipy import interpolate


class Interpolator:
    _FORT81_MAIN_CSV = 'fort.81.main.csv'
    _CSV_LABELS = ('x', 'x^2*n(x)')

    @staticmethod
    def interpolate_spectra(output, working_dir, x_start=-15, x_end=10, num=500):
        '''
        Crawls working directory, reads each steady state and fits a spline.
            Parameters:
                output (str):           The aggregate plot file.
                working_dir (str):      The working directory.
                x_start (int):          Start value to evaluate a spline function. Default is -15.
                x_end (int):            End value to evaluate a spline function. Default is 10.
                num (int):              Number of points to evaluate a spline function. Default is 500.
        '''
        logger = logging.getLogger(__name__)

        err = 0
        output = os.path.join(working_dir, output)
        if os.path.exists(output):
            logger.warning('Plot {} exists. Removing...'.format(output))
            os.remove(output)

        out_dict = dict()
        x_n = np.linspace(x_start, x_end, num)
        out_dict['x'] = x_n

        for (root, dirs, files) in os.walk(working_dir, topdown=False):
            main = Interpolator._FORT81_MAIN_CSV
            if main in files:
                run_id = int(os.path.basename(root))
                main = os.path.join(root, main)

                try:
                    logger.debug(
                        'Reading {} for interpolation...'.format(main))
                    y_n = Interpolator.interpolate_spectrum(main, x_n)
                    out_dict['y_{}'.format(run_id)] = y_n
                except BaseException as e:
                    logger.error('Reading {}: {}'.format(main, e))
                    err = err + 1

        logger.debug('Storing dict in file {}...'.format(output))
        df = pd.DataFrame(out_dict)
        df.to_csv(output, index=False)
        return err

    @staticmethod
    def interpolate_spectrum(file, x_n):
        '''
        Reads a csv and interpolates the data. The values at which the interpolation is evaluated are given as input.
            Parameters:
                file (str):             The file with the data to interpolate.
                x_n (np.array):         The values to evaluate the interpolation.
        '''
        logger = logging.getLogger(__name__)

        df = pd.read_csv(file)
        x = df[Interpolator._CSV_LABELS[0]]
        y = df[Interpolator._CSV_LABELS[1]]
        logger.debug('Interpolating {}...'.format(file))
        tck = interpolate.splrep(x, y)

        y_n = interpolate.splev(x_n, tck, der=0)
        return y_n
