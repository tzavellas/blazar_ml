import logging
import numpy as np
import os
import pandas as pd
from scipy import interpolate


def clamp(val, smallest=-30, largest=0):
    return np.maximum(smallest, np.minimum(val, largest))


class Interpolator:
    _FORT81_MAIN_CSV = 'fort.81.main.csv'
    _CSV_LABELS = ('x', 'x^2*n(x)')

    @staticmethod
    def interpolate_spectra(working_dir, x_start=-15,
                            x_end=10, num=500, k=1, clamped=None):
        '''
        Crawls working directory, reads each steady state and fits a spline.
            Parameters:
                working_dir (str):      The working directory.
                x_start (int):          Start value to evaluate a spline function. Default is -15.
                x_end (int):            End value to evaluate a spline function. Default is 10.
                num (int):              Number of points to evaluate a spline function. Default is 500.
        '''
        logger = logging.getLogger(__name__)

        err = 0

        interpolated_dict = dict()
        x_n = np.linspace(x_start, x_end, num)
        interpolated_dict['x'] = x_n

        with os.scandir(working_dir) as it:
            for entry in it:
                if entry.is_dir():
                    run_id = -1
                    try:
                        run_id = int(os.path.basename(entry))
                    except ValueError:
                        continue
                    s = 'y_{}'.format(run_id)
                    main = os.path.join(entry, Interpolator._FORT81_MAIN_CSV)
                    if os.path.exists(main):
                        try:
                            logger.debug(
                                'Reading {} for interpolation...'.format(main))
                            y_n = Interpolator.interpolate_spectrum(
                                main, x_n, k)
                            if clamped is None:
                                interpolated_dict[s] = y_n
                            else:
                                interpolated_dict[s] = clamp(
                                    y_n, clamped['min'], clamped['max'])
                        except BaseException as e:
                            logger.error('Reading {}: {}'.format(main, e))
                            err = err + 1
                    else:
                        y_n = np.empty(len(x_n))
                        y_n[:] = np.NaN
                        interpolated_dict[s] = y_n
        return err, interpolated_dict

    @staticmethod
    def interpolate_spectrum(file, x_n, k):
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
        tck = interpolate.splrep(x, y, k=k)

        y_n = interpolate.splev(x_n, tck, der=0)
        return y_n
