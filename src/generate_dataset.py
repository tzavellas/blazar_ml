#!/usr/bin/python

from datetime import datetime
from multiprocessing import Pool
import subprocess
import argparse
import os
import pandas as pd
import re
import shutil
import sys
import plot_spectra

elapsed_key = 'elapsed_time'
success_key = 'success'
run_key = 'run'

# Set of input keys that normaly are given in scientific notation
keys_scientific = set(['tol', 'temperat', 'exlumth',
                      'x1', 'x2', 'xbr', 'extph0', ])

# Set of csv keys to ignore while creating input dictionary
ignored_keys = set([elapsed_key, run_key, success_key])


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


def create_output_directory(run_id, working_dir=os.getcwd()):
    '''
    Creates the output directory
        Parameters:
            run_id (str):                   The id of the output directory
            working_dir (str):              The working directory. Default is the current working directory
    '''
    output_dir = os.path.realpath(working_dir + '/{}'.format(int(run_id)))

    if os.path.exists(output_dir):  # remove output_dir if it exists
        shutil.rmtree(output_dir)
    try:
        os.makedirs(output_dir)
    except OSError as e:
        print("Error: {} : {}".format(output_dir, e.strerror))

    return output_dir


def create_program_link(executable_path, dest_dir):
    '''
    Create symbolic link in the destination directory.
        Parameters:
            executable_path (str):          Full path to the program executable.
            dest_dir (str):                 The destination directory.
    '''
    base_name = os.path.basename(executable_path)
    link = '{}/{}'.format(dest_dir, base_name)
    os.symlink(executable_path, link)
    return link


def create_input_dictionary(input_series):
    '''
    Creates a dictionary with the all the input parameters of the program
        Parameters:
            input_series (pandas.Series):   A subset of all possible input parameters.
    '''
    input_dict = inputs_template  # copy template

    for index, value in input_series.iteritems():
        if index not in ignored_keys:
            for row in input_dict:
                if index in row:  # replace values that exist in input_series
                    row[index] = value

    return input_dict


def create_program_input(input_series, working_dir, input_fname='code.inp'):
    '''
    Compiles the absolute path of the input file of the program and writes it to disk.
        Parameters:
            input_series (pandas.Series):   A subset of all possible input parameters.
            working_dir (str):              The working directory.
            input_fname (str):              The name of the input file of the program. Default is "code.inp".
    '''
    # input file path name
    input_file = os.path.realpath(working_dir + '/{}'.format(input_fname))

    # open file
    with open(input_file, 'w') as f:
        input_dict = create_input_dictionary(input_series)
        values = ''
        keys = ''
        for row in input_dict:
            for key, value in row.items():
                keys = keys + ' {}'.format(key)
                if key in keys_scientific:
                    # print exponential with 2 decimal
                    values = values + ' {:.2e}'.format(value)
                else:
                    values = values + ' {}'.format(value)
            keys = keys + '\n'
            values = values + '\n'
        print(values, file=f) # print values at the begining of the file
        print(keys, file=f)   # print keys at the end of the file

    return input_file


def launch_process(executable, input_file, extra_args, output_dir):
    '''
    Launches a program instance. Program stdout is stored in a file "stdout.txt".
        Parameters:
            executable (str):       Program executable.
            input_file (str):       Program input file.
            extra_args (str):       Program extra arguments.
            output_dir (str):       Program output directory.
    '''
    cmd_args = [executable, input_file]     # Create shell command
    for arg in extra_args:
        cmd_args.append(arg)                # Appends extra_args to command, if any

    try:
        os.chdir(output_dir)                # Change to output directory

        with open('stdout.txt', 'w') as f:
            stream = subprocess.run(cmd_args, capture_output=True).stdout
            stream = stream.decode("utf-8")
            print(stream, file=f)           # Store stdout in a file for future reference
    except OSError as e:
        print("Error: {} : {}".format(output_dir, e.strerror))

    return stream


def parse_stream(stream, id):
    '''
    Parses stdout. Detects unsuccessful execution and extracts execution time.
        Parameters:
            stream (str):                   Program stdout stream.
            id (int):                       Run id of the current execution.
    '''
    overflow = stream.find('overflow!!!')           # search stream for overflow
    integration_fail = stream.find('IFAIL=2')       # search stream for IFAIL
    if (overflow == -1) and (integration_fail == -1):
        success = True
    else:
        print('Run {} failed'.format(id), file=sys.stderr)
        success = False
    
    pattern = 'Elapsed CPU time\D+(\d+\.\d+)'       # regex pattern
    match = re.search(pattern, stream)              # Find pattern
    if match:
        elapsed_time = float(match.group(1))
    else:
        print('Elapsed time pattern not found', file=sys.stderr)
        elapsed_time = .0

    return success, elapsed_time


def run_scenario(executable_path, input_series, id, working_dir, img_format, extra_args):
    '''
    Launches a new process.
        Parameters:
            executable_path (str):          Full path to the program executable.
            input_series (pandas.series):   The input file parameters of the program.
            id (int):                       Run id.
            working_dir (str):              The working directory.
            img_format (str):               Image format of the spectra plots.
            extra_args (list):              List of extra program arguments.
    '''
    out_dir = create_output_directory(id, working_dir)
    
    link = create_program_link(executable_path, out_dir)
    
    program_input = create_program_input(input_series, out_dir)

    out_stream = launch_process(link, program_input, extra_args, out_dir)

    success, elapsed_time = parse_stream(out_stream, id)

    if success:
        plot_spectra.save(id, out_dir, img_format)

    return id, success, elapsed_time


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
    parser.add_argument('-n', '--num-proc', type=int, default=None,
                        help='Number of processes to launch. Default is number of system threads')
    parser.add_argument('--overwrite-input', action='store_true', default=False,
                        help='Overwrites input CSV. Default is false')
    parser.add_argument('-x', '--extra-args', default=[], nargs='*',
                        help='List of extra arguments to pass to the program. Default is [].')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as arg_e:
        print('parse_args: {}'.format(arg_e), file=sys.stderr)
        sys.exit(1)

    exec_path = args.executable
    if not os.path.exists(exec_path):
        print('Executable {} does not exist'.format(exec_path), file=sys.stderr)
        sys.exit(1)

    try:
        inputs = pd.read_csv(args.input)
        inputs.insert(1, success_key, "False")       # add two extra columns
        inputs.insert(2, elapsed_key, 0.0)
    except BaseException as e:
        print('read_csv: {}'.format(e), file=sys.stderr)
        sys.exit(1)

    params = []
    for row in range(inputs.shape[0]):
        params.append((
            exec_path, 
            inputs.iloc[row], 
            row,
            args.working_dir, 
            args.format, 
            args.extra_args))

    start_time = datetime.now()

    with Pool(processes=args.num_proc) as pool:
        result = pool.starmap(run_scenario, params)
    
    end_time = datetime.now()
    
    print('Duration: {}'.format(end_time - start_time))  # prints total runtime duration

    for row, success, elapsed_time in result:
        inputs.at[row, success_key] = success
        inputs.at[row, elapsed_key] = elapsed_time

    if args.overwrite_input:
        out_csv = args.input
    else:
        basename = os.path.basename(args.input).split('.')[0]
        dirname = os.path.dirname(args.input)
        out_csv = '{}/{}_extended.csv'.format(dirname, basename)
        
    inputs.to_csv(out_csv, index=None)

    sys.exit(0)
