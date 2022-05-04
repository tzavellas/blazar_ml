#!/usr/bin/python

from multiprocessing import Pool
import subprocess
import argparse
import os
import pandas as pd
import shutil
import sys
import numpy as np
 
# Set of input keys that normaly are given in scientific notation
keys_scientific = set(['tol', 'temperat', 'exlumth',
                      'x1', 'x2', 'xbr', 'extph0', ])


# Dictionary with all possible program inputs. Keys with value None must be defined.
inputs_template = [
    {'ireadin': 0, 'npdec': 10, 'nsteps': 5, 'nout': 10, 'tmax': 5.},
    {'gpexmx': 10., 'ypmin': -6., 'yemin': -40., 'ygmin': -58., 'tol': 1e-4},
    {'slpinj': 2., 'slninj': 2., 'sleinj': 2.5, 'slginj': 2., 'slntinj': 2.},
    {'radius': None, 'bfield': None},
    {'iprext': 0, 'gpextmn': 5.21, 'gpextmx': 5.41, 'slprints': 2.01, 'exlumpr': -1.6, 'bpresc': 1., 'ap': 0., 'ipexp': 0},
    {'ielext': 1, 'geextmn': None, 'geextmx': None, 'slelints': None, 'exlumel': None, 'belesc': 1., 'ae': 0., 'ieexp': 0},
    {'iphotext': 0, 'temperat': 1e6, 'exlumth': 8.1e-6},
    {'iphotext2': 0, 'x1': 1e-4, 'x2': 1.5e-4, 'xbr': 1.5e-4, 'beta1': 2., 'beta2': 2., 'extph0': 1e-1},
    {'ielextbr': 0, 'geextbr': 4.01, 'slelints1': 1.6, 'slelints2': 4.5, 'ae2': 1.},
    {'iprextbr': 0, 'gpextbr': 5.01, 'slprints1': 1.6, 'slprints2': 2.6, 'ap2': 1.},
    {'isyn': 1, 'iprsyn': 0, 'imsyn': 0, 'ipsyn': 0, 'iksyn': 0, 'issa': 1},
    {'icompt': 1, 'ikn': 0, 'igg': 0, 'ianni': 0, 'iesc': 1, 'ipsc': 1},
    {'ipair': 0, 'ipion': 0, 'ineutron': 0},
]

def create_input_file(inputs, working_dir='./', input_file='code.inp'):
    '''
    Compiles the absolute file name and writes it
            Parameters:
                    inputs (dict): all the input parameters of the program
                    working_dir (str): The working directory, default is ./
                    input_file (float): The input file of the program
    '''
    # input file path name
    file_path = os.path.realpath(working_dir + '/{}'.format(input_file))

    # open file
    with open(file_path, 'w') as f:
        for row in inputs:
            #line = ''  # each row in a different line for readability
            line = []  # each row in a different line for readability
            for key, value in row.items():
                if key in keys_scientific:
                    # print exponential with 2 decimal
                    line.append(value)                     
                else:
                    #line = line + ' {}'.format(value)
                    line.append(value)  
            print(*line, file=f)
    f.close()

    return file_path


def init_input_dict(input_series):
    '''
    Creates a dictionary with the all the input parameters of the program
            Parameters:
                    input_series (pandas.Series): Values of a subset of the input parameters
    '''
    input_dict = inputs_template  # copy template

    for index, value in input_series.iteritems():
        if index != 'run':        # ignore the key run
            for row in input_dict:
                if index in row:  # replace values that exist in input_series
                    row[index] = value

    return input_dict


def create_output_directory(run_id, working_dir='./'):
    '''
    Creates the output directory
            Parameters:
                    run_id (str): The id of the output directory
                    working_dir (str): The working directory, default is ./
    '''
    output_dir = os.path.realpath(working_dir + '/{}'.format(int(run_id)))

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # remove output_dir if it exists
    try:
        os.makedirs(output_dir)
    except OSError as e:
        print("Error: {} : {}".format(output_dir, e.strerror))

    return output_dir


def execute_input(input_file, output_dir):
    '''
    Launches a new process.
            Parameters:
                    input_file (str): The input file parameters of the program
                    output_dir (str): The program output directory
    '''
    # Create shell command
    cmd_args = ['echo', input_file]

    try:        
        os.chdir(output_dir)
        output = subprocess.run(cmd_args, capture_output=True).stdout
        print(output)
    except OSError as e:
        print("Error: {} : {}".format(output_dir, e.strerror))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generates the dataset given a sample of the input space')
    parser.add_argument('--input', required=True, type=str,
                        help='CSV file with the a sample of the input space')
    parser.add_argument('--working-dir', default='output',
                        type=str, help='Root path where the datase will be stored')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as arg_e:
        print('parse_args: {}'.format(arg_e))
        sys.exit(1)

    try:
        inputs = pd.read_csv(args.input)
        params = []
        for row in range(inputs.shape[0]):
            output_dir = create_output_directory(row, args.working_dir)

            input_dict = init_input_dict(inputs.iloc[row])

            input_file = create_input_file(input_dict, output_dir)

            params.append((input_file, output_dir))

        with Pool(processes=None) as pool:
            pool.starmap(execute_input, params)

    except BaseException as e:
        print('read_csv: {}'.format(e))
        sys.exit(1)

    sys.exit(0)
