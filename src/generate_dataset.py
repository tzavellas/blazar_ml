#!/usr/bin/python

from datetime import datetime
from multiprocessing import Pool
import argparse
import logging
import os
import pandas as pd
import re
import shutil
import subprocess
import sys
import plot_spectra

# Script constants
code_inp = 'code.inp'
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
    {'gpexmx': 8., 'ypmin': -6., 'yemin': -40., 'ygmin': -60., 'tol': 1e-4},
    {'slpinj': 2., 'slninj': 2., 'sleinj': 3.0, 'slginj': 2., 'slntinj': 2.},
    {'radius': None, 'bfield': None},
    {'iprext': 0, 'gpextmn': 5.21, 'gpextmx': 5.41, 'slprints': 2.01, 'exlumpr': -1.6, 'bpresc': 1., 'ap': 0., 'ipexp': 0},
    {'ielext': 1, 'geextmn': None, 'geextmx': None, 'slelints': None, 'exlumel': None, 'belesc': 1., 'ae': 0., 'ieexp': 0},
    {'iphotext': 0, 'temperat': 1e6, 'exlumth': 8.1e-6},
    {'iphotext2': 0, 'x1': 1e-4, 'x2': 1.5e-4, 'xbr': 1.5e-4, 'beta1': 2., 'beta2': 2., 'extph0': 1e-1},
    {'ielextbr': 0, 'geextbr': 4.01, 'slelints1': 1.6, 'slelints2': 4.5, 'ae2': 1.},
    {'iprextbr': 0, 'gpextbr': 5.01, 'slprints1': 1.6, 'slprints2': 2.6, 'ap2': 1.},
    {'isyn': 1, 'iprsyn': 0, 'imsyn': 0, 'ipsyn': 0, 'iksyn': 0, 'issa': 0},
    {'icompt': 1, 'ikn': 0, 'igg': 0, 'ianni': 0, 'iesc': 1, 'ipsc': 1},
    {'ipair': 0, 'ipion': 0, 'ineutron': 0},
]

def create_output_directory(run_id, working_dir, logger):
    '''
    Creates the output directory
        Parameters:
            run_id (str):                   The id of the output directory.
            working_dir (str):              The working directory.
    '''
    output_dir = os.path.realpath(working_dir + '/{}'.format(int(run_id)))

    if os.path.exists(output_dir):  # remove output_dir if it exists
        logger.debug('Dir {} exists. Removing...'.format(output_dir))
        shutil.rmtree(output_dir)
    try:
        logger.debug('Creating dir {}'.format(output_dir))
        os.makedirs(output_dir)
    except OSError as e:
        logger.error('Error: {} : {}'.format(output_dir, e.strerror))

    return output_dir


def create_program_link(executable_path, dest_dir, logger):
    '''
    Create symbolic link in the destination directory.
        Parameters:
            executable_path (str):          Full path to the program executable.
            dest_dir (str):                 The destination directory.
    '''
    base_name = os.path.basename(executable_path)
    link = '{}/{}'.format(dest_dir, base_name)
    logger.debug('Creating symlink {}'.format(link))
    os.symlink(executable_path, link)
    return link


def create_input_dictionary(input_series, logger):
    '''
    Creates a dictionary with the all the input parameters of the program
        Parameters:
            input_series (pandas.Series):   A subset of all possible input parameters.
    '''
    input_dict = inputs_template  # copy template

    logger.debug('Creating input dictionary...')

    for index, value in input_series.iteritems():
        if index not in ignored_keys:
            for row in input_dict:
                if index in row:  # replace values that exist in input_series
                    row[index] = value

    return input_dict


def create_program_input(input_series, working_dir, input_fname, logger):
    '''
    Compiles the absolute path of the input file of the program and writes it to disk.
        Parameters:
            input_series (pandas.Series):   A subset of all possible input parameters.
            working_dir (str):              The working directory.
            input_fname (str):              The name of the input file of the program.
    '''
    # input file path name
    input_file = os.path.realpath(working_dir + '/{}'.format(input_fname))

    # open file
    logger.debug('Open to write {}'.format(input_file))

    with open(input_file, 'w') as f:
        input_dict = create_input_dictionary(input_series, logger)
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

    logger.debug('{} written'.format(input_file))

    return input_file


def launch_process(executable, input_file, extra_args, output_dir, id, logger):
    '''
    Launches a program instance. Program stdout is stored in a file "stdout.txt".
        Parameters:
            executable (str):       Program executable.
            input_file (str):       Program input file.
            extra_args (str):       Program extra arguments.
            output_dir (str):       Program output directory.
            id (int):               Run id of the current execution.
    '''
    cmd_args = [executable, input_file]     # Create shell command
    for arg in extra_args:
        cmd_args.append(arg)                # Appends extra_args to command, if any
    logger.debug('Run {} command: {}'.format(id, cmd_args))

    try:
        stream_file = '{}/stdout.txt'.format(output_dir)
        logger.debug('Run {} subprocess stdout file {}'.format(id, stream_file))

        with open(stream_file, 'w') as f:
            logger.debug('Run {} subprocess launch'.format(id))
            stream = subprocess.run(cmd_args, capture_output=True, cwd=output_dir).stdout
            stream = stream.decode("utf-8")
            print(stream, file=f)           # Store stdout in a file for future reference

        logger.debug('Run {} subprocess finished'.format(id))

    except OSError as e:
        logger.error('Run {} error: {} : {}'.format(id, output_dir, e.strerror))

    return stream


def parse_stream(stream, id, logger):
    '''
    Parses stdout. Detects unsuccessful execution and extracts execution time.
        Parameters:
            stream (str):                   Program stdout stream.
            id (int):                       Run id of the current execution.
    '''
    overflow = stream.find('overflow!!!')           # search stream for overflow
    logger.debug('Run {} search overflow: {}'.format(id, overflow))

    integration_pattern = 'IFAIL=\s*2'              # search stream for IFAIL
    integration_fail = re.search(integration_pattern, stream)
    logger.debug('Run {} search integration failure: {}'.format(id, integration_fail))

    if (overflow == -1) and (integration_fail is None):
        success = True
    else:
        logger.error('Run {} failed'.format(id))
        success = False
    
    pattern = 'Elapsed CPU time\D+(\d+\.\d+)'       # regex pattern
    match = re.search(pattern, stream)              # Find pattern

    if match:
        elapsed_time = float(match.group(1))
        logger.debug('Run {} search elapsed CPU time: {}'.format(id, elapsed_time))
    else:
        logger.error('Run {} Elapsed time pattern not found'.format(id))
        elapsed_time = .0

    return success, elapsed_time


def run_scenario(executable_path, input_series, id, working_dir, img_format, extra_args, logger):
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
    out_dir = create_output_directory(id, working_dir, logger)
    
    link = create_program_link(executable_path, out_dir, logger)
    
    program_input = create_program_input(input_series, out_dir, code_inp, logger)

    out_stream = launch_process(link, program_input, extra_args, out_dir, id, logger)

    success, elapsed_time = parse_stream(out_stream, id, logger)

    logger.info('Run {} parse success: {}'.format(id, success))
    if success:
        logger.info('Run {} saving spectrum in: {}'.format(id, out_dir))
        plot_spectra.save(id, out_dir, img_format)

    return id, success, elapsed_time


def init_logger(logfile):
    logging.getLogger('matplotlib').disabled = True
    logging.getLogger('matplotlib.font_manager').disabled = True
    logger = logging.getLogger()

    formatter= logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    
    file_handler = logging.FileHandler(filename=logfile, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.setLevel(logging.DEBUG)
    return logger


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
    parser.add_argument('-l', '--logging', type=str, default='generate_dataset.log',
                        help='Log file. Default is generate_dataset.log.')
    parser.add_argument('-n', '--num-proc', type=int, default=None,
                        help='Number of processes to launch. Default is number of system threads')
    parser.add_argument('-r','--overwrite-input', action='store_true', default=False,
                        help='Overwrites input CSV. Default is false')
    parser.add_argument('-p', '--plot-spectra', action='store_true', default=True,
                        help='Aggregate plot of all spectra in a file "spectra.png". Default is True.')
    parser.add_argument('-x', '--extra-args', default=[], nargs='*',
                        help='List of extra arguments to pass to the program. Default is [].')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as arg_e:
        print('parse_args: {}'.format(arg_e), file=sys.stderr)
        sys.exit(1)

    logfile = os.path.abspath(args.logging)
    if os.path.exists(logfile):
        os.remove(logfile)
    print('Logfile: {}'.format(logfile))

    logger = init_logger(logfile)
        
    logger.debug('check executable')
    exec_path = os.path.abspath(args.executable)
    if not os.path.exists(exec_path):
        logger.error('Executable {} does not exist'.format(exec_path))
        sys.exit(1)

    logger.debug('check input file')
    input = os.path.abspath(args.input)
    if not os.path.exists(input):
        logger.error('Input file {} does not exist'.format(input))
        sys.exit(1)

    logger.debug('check working dir')
    working_dir = os.path.abspath(args.working_dir)
    if not os.path.exists(working_dir):
        logger.error('working dir {} does not exist'.format(input))
        os.makedirs(working_dir)
    else:
        shutil.rmtree(working_dir, ignore_errors=True)

    try:
        logger.debug('read_csv '.format(input))
        inputs = pd.read_csv(input)
        inputs.insert(1, success_key, "False")       # add two extra columns
        inputs.insert(2, elapsed_key, 0.0)
    except BaseException as e:
        logger.error('read_csv: {}'.format(e))
        sys.exit(1)

    logger.info('build scenario parameters')
    params = []
    for row in range(inputs.shape[0]):
        params.append((
            exec_path, 
            inputs.iloc[row], 
            row,
            args.working_dir, 
            args.format, 
            args.extra_args,
            logger))

    start_time = datetime.now()

    with Pool(processes=args.num_proc) as pool:
        result = pool.starmap(run_scenario, params)
    
    end_time = datetime.now()
    
    logger.info('Duration: {}'.format(end_time - start_time)) # logs total runtime duration

    for row, success, elapsed_time in result:
        inputs.at[row, success_key] = success
        inputs.at[row, elapsed_key] = elapsed_time
    
    if args.overwrite_input:
        out_csv = input
    else:
        basename = os.path.basename(input).split('.')[0]    # basename without extension
        dirname = os.path.dirname(input)                    # directory name
        out_csv = '{}/{}_extended.csv'.format(dirname, basename)
        # if os.path.exists(out_csv):
        #     os.remove(out_csv)
    
    logger.info('write result to csv {}'.format(out_csv))
    inputs.to_csv(out_csv, index=None)

    if args.plot_spectra:
        spectra = '{}/spectra'.format(args.working_dir)
        err = plot_spectra.aggregate_plots(spectra, args.working_dir, args.format, True, logger)
        if  err != 0:
            logger.error('Error plotting aggregate spectrum {}'.format(spectra))

    sys.exit(err)
