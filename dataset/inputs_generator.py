import copy
import logging


class InputsGenerator:
    # Default input parameters
    _DEFAULT_INPUT_PARAMS = [
        {'ireadin': 0, 'npdec': 10, 'nsteps': 5, 'nout': 10, 'tmax': 5.},
        {'gpexmx': 8., 'ypmin': -6., 'yemin': -40., 'ygmin': -60., 'tol': 1e-4},
        {'slpinj': 2., 'slninj': 2., 'sleinj': 3.0, 'slginj': 2., 'slntinj': 2.},
        {'radius': None, 'bfield': None},
        {'iprext': 0, 'gpextmn': 5.21, 'gpextmx': 5.41, 'slprints': 2.01,
            'exlumpr': -1.6, 'bpresc': 1., 'ap': 0., 'ipexp': 0},
        {'ielext': 1, 'geextmn': None, 'geextmx': None, 'slelints': None,
            'exlumel': None, 'belesc': 1., 'ae': 0., 'ieexp': 0},
        {'iphotext': 0, 'temperat': 1e6, 'exlumth': 8.1e-6},
        {'iphotext2': 0, 'x1': 1e-4, 'x2': 1.5e-4, 'xbr': 1.5e-4,
            'beta1': 2., 'beta2': 2., 'extph0': 1e-1},
        {'ielextbr': 0, 'geextbr': 4.01, 'slelints1': 1.6,
            'slelints2': 4.5, 'ae2': 1.},
        {'iprextbr': 0, 'gpextbr': 5.01, 'slprints1': 1.6,
            'slprints2': 2.6, 'ap2': 1.},
        {'isyn': 1, 'iprsyn': 0, 'imsyn': 0, 'ipsyn': 0, 'iksyn': 0, 'issa': 0},
        {'icompt': 1, 'ikn': 0, 'igg': 0, 'ianni': 0, 'iesc': 1, 'ipsc': 1},
        {'ipair': 0, 'ipion': 0, 'ineutron': 0},
    ]

    # Keys that are ignored while setting input parameters
    _IGNORED_KEYS = ['run', 'success', 'elapsed_time']

    # Keys that are printed with scientific notation
    _SCIENTIFIC_KEYS = set(
        ['tol', 'temperat', 'exlumth', 'x1', 'x2', 'xbr', 'extph0', ])

    def __init__(self, input_params={}):
        self._input_params = copy.deepcopy(self._DEFAULT_INPUT_PARAMS)
        self.input_params = input_params

    def to_file(self, file):
        '''
        Writes the parameters in a file.
            Parameters:
                file (str):   The file path.
        '''
        with open(file, 'w') as f:
            logger = logging.getLogger(__name__)
            s = self.to_str()
            logger.info('Writing {}'.format(file))
            f.write(s)

    def to_str(self):
        '''
        Writes the parameters in a string and returns it.
            Parameters:
        '''
        values = ''
        keys = ''
        for row in self.input_params:
            for key, value in row.items():
                keys = keys + ' {}'.format(key)
                if key in self._SCIENTIFIC_KEYS:
                    # print exponential with 2 decimal
                    values = values + ' {:.2e}'.format(value)
                else:
                    values = values + ' {}'.format(value)
            keys = keys + '\n'
            values = values + '\n'
        # write values, new line and then keys
        ret = '{}\n\n{}'.format(values, keys)
        return ret

    def get_input_parameters(self):
        '''
        Gets a dictionary of the parameters
        '''
        return self._input_params

    def set_input_parameters(self, input):
        '''
        Sets the values of the parameters given in a dictionary
            Parameters:
                input (dict):   The dictionary.
        '''
        # Remove ignored keys, e.g. "run"
        for key in self._IGNORED_KEYS:
            input.pop(key, None)

        logger = logging.getLogger(__name__)
        # replace values that exist in input
        for key, value in input.items():
            for row in self._input_params:
                if key in row:
                    logger.debug('Updating {} to {}'.format(key, value))
                    row[key] = value
                    break

    input_params = property(get_input_parameters, set_input_parameters)
