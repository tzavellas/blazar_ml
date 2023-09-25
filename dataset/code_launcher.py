import errno
from inputs_generator import InputsGenerator
from plotter import Plotter
from stream_parser import parse_stream
import logging
import os
import pandas as pd
import shutil
import subprocess


def to_main(string):
    return '{}.main'.format(string)


def to_baseline(string):
    return '{}.baseline'.format(string)


class CodeLauncher:
    _CODE_INP = 'code.inp'
    _FORT81 = 'fort.81'
    _EXTENDED_CSV = 'extended.csv'
    _PLOT = 'plot_clean'
    ELAPSED_TIME_KEY = 'elapsed_time'
    ELAPSED_TIME_BASELINE_KEY = 'elapsed_time_baseline'
    SUCCESS_KEY = 'success'
    _CSV_LABELS = ('x', 'x^2*n(x)')

    def __init__(self, exec_path='', working_dir='',
                 input_df='', img_format='png'):
        self.exec_path = exec_path
        self.working_dir = working_dir
        self.inputs_dataframe = input_df
        self._img_format = img_format

    def __get_input_at(self, row):
        '''
        Convenience function. Retrieves a given row of the inputs dataframe,
        converts it to dictionary and returns it.
            Parameters:
                output_dir (str):               The output directory.
        '''
        if row < len(self.inputs_dataframe.index):
            input_dict = self.inputs_dataframe.iloc[row].to_dict()
        return input_dict

    def __run_case(self, input_snapshot, exec_path, output_dir, extra_args):
        '''
        Convenience function. Runs the program with inputs from the id-th row.
            Parameters:
                row (int):                      The row-th case from the inputs dataframe.
                exec_path (str):                The executable path.
                output_dir (str):               The program output directory.
                extra_args (tuple):             Extra arguments to pass to the program.
        '''

        # create program input
        program_input = self.create_program_input(
            input_snapshot, output_dir, self._CODE_INP)
        # launch program as a separate process
        out_stream = self.launch_process(
            exec_path, program_input, output_dir, 'stdout.txt', extra_args)
        # parse stdout for errors, elapsed time etc
        success, elapsed_time = parse_stream(out_stream)

        spectrum, x, y = '', [], []
        if success:
            src = self._CODE_INP
            dst = to_main(self._CODE_INP)
            CodeLauncher.mv(src, dst, output_dir)
            spectrum, x, y = CodeLauncher.save(output_dir, baseline=False)

        src = self._FORT81
        dst = to_main(self._FORT81)
        CodeLauncher.mv(src, dst, output_dir)

        return success, elapsed_time, x, y, input_snapshot

    def __run_baseline(self, input_snapshot, exec_path,
                       output_dir, extra_args):
        '''
        Convenience function. Runs the baseline case.
            Parameters:
                input_snapshot (dict):          TODO
                exec_path (str):                The executable path.
                output_dir (str):               The program output directory.
                extra_args (tuple):             Extra arguments to pass to the program.
        '''
        # create program input
        program_input = self.create_program_input(
            input_snapshot, output_dir, self._CODE_INP)
        # launch program as a separate process
        out_stream = self.launch_process(
            exec_path, program_input, output_dir, 'stdout.baseline.txt', extra_args)
        # parse stdout for errors, elapsed time etc
        success, elapsed_time = parse_stream(out_stream)

        spectrum, x, y = '', [], []
        if success:
            src = self._CODE_INP
            dst = to_baseline(self._CODE_INP)
            CodeLauncher.mv(src, dst, output_dir)
            spectrum, x, y = CodeLauncher.save(output_dir, baseline=True)

        src = self._FORT81
        dst = to_baseline(self._FORT81)
        CodeLauncher.mv(src, dst, output_dir)

        return success, elapsed_time, x, y

    @staticmethod
    def extract_steady_state(file):
        '''
        Reads an fort.81 file and extracts the steady state values.
            Parameters:
                file (str) : The fort.81 file
        '''
        x, y = [], []
        with open(file) as f:
            for line in f:                      # read line by line
                tokens = line.rstrip().split()  # tokenize line
                if len(tokens) == 4:            # new block starts
                    x.clear()
                    y.clear()
                    continue
                else:                           # same block
                    x.append(float(tokens[0]))
                    y.append(float(tokens[1]))
        return x, y

    @staticmethod
    def mv(src, dst, working_dir):
        '''
        Wrapper function. Renames a file in a working dir.
            Parameters:
                src (str):                      The source file.
                dst (str):                      The destination file.
                working_dir (str):              The working directory.
        '''
        logger = logging.getLogger(__name__)

        s = os.path.join(working_dir, src)
        d = os.path.join(working_dir, dst)
        logger.debug('Rename {} to {}'.format(s, d))
        os.rename(s, d)

    @staticmethod
    def to_file(x, y, file, labels=_CSV_LABELS):
        '''
        Saves x and y in a csv file.
            Parameters:
                x (list):               x-axis values
                y (list):               y-axis values
                file (str):             File path to save the data.
                labels (tuple):         The labels of the csv columns. Default x -> "x",  y -> "x^2*n(x)"
        '''
        df = pd.DataFrame()
        df[labels[0]] = x
        df[labels[1]] = y
        df.to_csv(file, index=False, float_format='%.6e', lineterminator='\n')
        return

    @staticmethod
    def plot(x, y, working_dir, img_format='png'):
        '''
        Wrapper function. Plots x,y values.
            Parameters:
                x (list):               x-axis values
                y (list):               y-axis values
                working_dir (str):      Directory where the fort.81 exists.
                img_format (bool):      Image format of the plot. Default is png.
        '''
        logger = logging.getLogger(__name__)

        id = int(os.path.basename(working_dir))
        logger.debug('Plot clean spectrum {}'.format(id))
        plot_file = os.path.join(
            working_dir, '{}.{}'.format(CodeLauncher._PLOT, img_format))
        Plotter.plot_spectrum(x, y, id, file=plot_file)
        logger.debug('Plot clean spectrum {} finished'.format(id))

    @staticmethod
    def save(working_dir, baseline=False):
        '''
        Wrapper function. Reads a fort.81 file in the working directory, extracts
        the steady state spectrum, saves it in a CSV file. Optionally, it saves a
        plot of the spectrum
            Parameters:
                working_dir (str):      Directory where the fort.81 exists.
                baseline (bool):        Enables saving the baseline spectrum (i.e. without injections).
        '''
        logger = logging.getLogger(__name__)

        fort81 = os.path.join(working_dir, CodeLauncher._FORT81)

        fort81_ss = to_baseline(fort81) if baseline else to_main(fort81)
        fort81_ss = '{}.csv'.format(fort81_ss)

        x, y = [], []
        if os.path.exists(fort81):
            logger.info('Saving steady state {}'.format(fort81_ss))
            x, y = CodeLauncher.extract_steady_state(fort81)
            CodeLauncher.to_file(x, y, file=fort81_ss)
        else:
            logger.error('File {} does not exist'.format(fort81))

        return fort81_ss, x, y

    @staticmethod
    def write_dataframe(input_dataframe, output_dir):
        '''
        Writes the dataframe to a file on the disk under a given output directory.
        The file name is given by the _EXTENDED_CSV variable.
            Parameters:
                output_dir (str):               The output directory.
        '''
        extended = os.path.join(output_dir, CodeLauncher._EXTENDED_CSV)
        input_dataframe.to_csv(extended, index=None)

    def get_exec_path(self):
        '''
        Getter for the executable path.
        '''
        return self._exec_path

    def get_working_dir(self):
        '''
        Getter for the working directory.
        '''
        return self._working_dir

    def get_inputs_dataframe(self):
        '''
        Getter for the inputs dataframe.
        '''
        return self._inputs_dataframe

    def load_inputs_dataframe(self, file=None):
        '''
        Load inputs dataframe

        Parameters
        ----------
        file : str
            Inputs file path.

        Returns
        -------
        bool
            True if file exists and load was successful.

        '''
        logger = logging.getLogger(__name__)
        if file is None:
            file = os.path.join(self._working_dir, self._EXTENDED_CSV)
        if os.path.exists(file):
            self._inputs_dataframe = pd.read_csv(file)
            logger.info('Inputs dataframe {}'.format(file))
            return True

    def set_exec_path(self, exec_path):
        '''
        Setter for the executable path.
            Parameters:
                exec_path (str):            The executable path.
        '''
        logger = logging.getLogger(__name__)

        if os.path.isfile(exec_path):
            if os.path.isabs(exec_path):
                self._exec_path = exec_path
            else:
                self._exec_path = os.path.abspath(exec_path)
            logger.info('Executable path {}'.format(self._exec_path))
        else:
            self._exec_path = ''
            logger.warning('No executable path')

    def set_working_dir(self, dir):
        '''
        Setter for the working directory.
            Parameters:
                dir (str):                  The working directory.
        '''
        logger = logging.getLogger(__name__)

        if os.path.isabs(dir):
            self._working_dir = dir
        else:
            self._working_dir = os.path.abspath(dir)
        logger.info('working directory {}'.format(self._working_dir))

    def set_inputs_dataframe(self, input_file):
        '''
        Setter for the inputs dataframe.
            Parameters:
                input_file (str):           The csv file containing the inputs.
        '''
        logger = logging.getLogger(__name__)

        if not os.path.isabs(input_file):
            input_file = os.path.join(self.working_dir, input_file)

        if self.load_inputs_dataframe(input_file):
            # add two extra columns
            self._inputs_dataframe.insert(1, self.SUCCESS_KEY, "False")
            self._inputs_dataframe.insert(2, self.ELAPSED_TIME_KEY, 0.0)
            self._inputs_dataframe.insert(
                3, self.ELAPSED_TIME_BASELINE_KEY, 0.0)
            logger.debug('Added columns: {}, {}, {}'.format(
                self.SUCCESS_KEY, self.ELAPSED_TIME_KEY, self.ELAPSED_TIME_BASELINE_KEY))
        else:
            self._inputs_dataframe = None
            logger.debug(
                'Inputs file {} does not exist. No inputs dataframe is set'.format(input_file))

    exec_path = property(get_exec_path, set_exec_path)
    working_dir = property(get_working_dir, set_working_dir)
    inputs_dataframe = property(get_inputs_dataframe, set_inputs_dataframe)

    def create_program_input(self, input_snapshot, dest_dir, file):
        '''
        Compiles the absolute path of the input file of the program and writes it to disk.
            Parameters:
                dest_dir (str):                 The path of the input file.
                input_file (str):               The input file name.
        '''
        # create program input file path
        program_input = os.path.join(dest_dir, file)
        # write to file
        input_snapshot.to_file(program_input)

        return program_input

    def mkdir(self, directory):
        '''
        Creates a directory under the working directory and returns its path.
        If the directory already exists, it removes it along with its contents.
            Parameters:
                working_dir (str):              The working directory.
        '''
        logger = logging.getLogger(__name__)

        output_dir = os.path.join(self.working_dir, str(directory))

        if os.path.exists(output_dir):  # remove output_dir if it exists
            logger.warning(
                'Output directory {} already exists. Removing...'.format(output_dir))
            shutil.rmtree(output_dir)

        logger.info('Creating output directory {}'.format(output_dir))
        os.makedirs(output_dir)

        return output_dir

    def ln(self, dest_dir=None):
        '''
        Create symbolic link of the executable in the destination directory.
            Parameters:
                dest_dir (str):                 The destination directory.
        '''
        logger = logging.getLogger(__name__)

        if not os.path.exists(self._exec_path):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), self._exec_path)

        if dest_dir is None:
            dest_dir = self._working_dir
            logger.debug('no destination dir. setting {}'.format(dest_dir))

        base_name = os.path.basename(self._exec_path)
        link = os.path.join(dest_dir, base_name)

        if os.path.exists(link):
            logger.debug('Link {} already exists. Removing...'.format(link))
            os.remove(link)

        logger.info('Creating symlink {}'.format(link))
        os.symlink(self.exec_path, link)
        return link

    def launch_process(self, executable, program_input,
                       output_dir, outstream_file, extra_args):
        '''
        Launches a program instance. Program stdout is stored in a file.
            Parameters:
                executable (str):       Program executable.
                program_input (str):    Program input file.
                output_dir (str):       Program output directory.
                outstream_file (str):   File that stores the stdout.
                extra_args (str):       Program extra arguments.
        '''
        logger = logging.getLogger(__name__)

        ret = ''
        if not os.path.exists(executable):
            logger.error('Executable {} does not exist'.format(executable))
            return ret

        if not os.path.exists(program_input):
            logger.error(
                'Program input {} does not exist. Aborting launch_process...'.format(program_input))
            return ret

        if not os.path.exists(output_dir):
            logger.warning('Process output dir not set.')
            output_dir = self.working_dir

        logger.debug('Process output dir {}'.format(output_dir))

        cmd_args = [executable, program_input]     # Create shell command
        for arg in extra_args:
            # Appends extra_args to command, if any
            cmd_args.append(arg)

        try:
            stream_file = os.path.join(output_dir, outstream_file)
            logger.info('Stdout will be written at {}'.format(stream_file))

            with open(stream_file, 'w') as f:
                logger.info('Launch process {}'.format(cmd_args))
                stream = subprocess.run(
                    cmd_args, capture_output=True, cwd=output_dir).stdout
                ret = stream.decode("utf-8")
                # Store stdout in a file for future reference
                f.write(ret)
                logger.info('Finished process {}'.format(cmd_args))

        except OSError as e:
            logger.error('Error {}'.format(e.strerror))

        return ret

    def run(self, run_id, extra_args=[], img_format='.png'):
        '''
        Runs the program for a given row of the input dataframe.
            Parameters:
                run_id (int):                   The row of the dataframe.
                extra_args (list):              List of extra program arguments.
                img_format (str):               Image format of the spectra plots.
        '''
        logger = logging.getLogger(__name__)

        # create id-th case output directory
        out_dir = self.mkdir(str(run_id))
        # create link in the output directory
        link = self.ln(out_dir)

        # get input dictionary from row
        input_dict = self.__get_input_at(run_id)
        # create an input snapshot
        input_snapshot = InputsGenerator(input_dict)
        logger.debug('run id {}, input snapshot {}'.format(
            run_id, hex(id(input_snapshot))))
        # run the case
        case_ret = self.__run_case(input_snapshot, link, out_dir, extra_args)

        # deactivate flags
        input_snapshot = case_ret[4]
        no_injection = {'ielext': 0}
        input_snapshot.set_input_parameters(no_injection)

        # run the baseline case
        baseline_ret = self.__run_baseline(
            input_snapshot, link, out_dir, extra_args)

        logger.debug('Store result')
        # Save success/elapsed time in the dataframe
        success = case_ret[0] and baseline_ret[0]
        elapsed_time = case_ret[1]
        elapsed_time_baseline = baseline_ret[1]

        # Clean spectrum and plot
        if success:
            logger.debug('Clean spectrum')
            actual = (case_ret[2], case_ret[3])
            baseline = (baseline_ret[2], baseline_ret[3])
            x, y = Plotter.diff(actual, baseline)
            CodeLauncher.plot(x, y, out_dir, img_format)

        return run_id, success, elapsed_time, elapsed_time_baseline
