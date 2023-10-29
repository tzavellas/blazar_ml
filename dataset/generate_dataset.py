import argparse
import json
from multiprocessing import Pool
import os
import pandas as pd
import logging.config
import sys
import timeit
from code_launcher import CodeLauncher
from dataset_creator import DatasetCreator
from interpolator import Interpolator
from plotter import Plotter


if __name__ == "__main__":
    filename = os.path.basename(__file__).split('.')[0]

    parser = argparse.ArgumentParser(
        description='Generates the dataset given a sample of the input space.')
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        required=True,
        help='Config file.')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as arg_e:
        print('parse_args: {}'.format(arg_e), file=sys.stderr)
        sys.exit(1)

    with open(args.config) as config:
        config = json.loads(config.read())

        paths = config['paths']
        options = config['options']

        input_path = paths.get('input', '')
        if not os.path.exists(input_path):
            print('Input csv {} does not exist'.format(input_path))
            sys.exit(1)

        working_dir = os.path.abspath(paths.get('working_dir', ''))
        if not os.path.exists(working_dir):
            print(
                'Working directory {} does not exist. Creating it..'.format(working_dir))
            os.mkdir(working_dir)

        skip = False
        executable = paths.get('executable', '')
        if not os.path.exists(executable):
            print('Executable {} does not exist'.format(executable))
            print('CodeLauncher will be skipped')
            skip = True

        logfile = os.path.join(working_dir, '{}.log'.format(filename))
        if os.path.exists(logfile):
            os.remove(logfile)

        try:
            logging_ini = paths.get('logging', 'logging.ini')
            print('Loading {} ...'.format(logging_ini))
            logging.config.fileConfig(logging_ini, defaults={
                                      'logfilename': '{}'.format(logfile)}, disable_existing_loggers=True)
            print('Logfile: {}'.format(logfile))
            logger = logging.getLogger(os.path.basename(__file__))
        except Exception as e:
            print(
                'Failed to load config from {}. Exception {}'.format(
                    logging_ini, e))
            logging.basicConfig(
                format='%(asctime)s %(name)s - %(levelname)s: %(message)s')
            logging.getLogger('matplotlib').disabled = True
            logging.getLogger('matplotlib.font_manager').disabled = True
            logger = logging.getLogger(os.path.basename(__file__))
            logger.setLevel(logging.DEBUG)

        launcher = CodeLauncher(exec_path=executable,
                                working_dir=working_dir,
                                input_df=input_path)

        rows = launcher.get_inputs_dataframe().shape[0]
        logger.info('Input csv contains {} rows'.format(rows))

        img_format = options.get('format', 'png')
        skip = skip or options.get('skip_run', False)

        if not skip:
            extra_args = options.get('extra_args', [])
            num_proc = options.get('num_proc', None)
            params = []
            for row in range(rows):
                params.append((row, extra_args, img_format))
            start_time = timeit.default_timer()
            # run in parallel processes
            with Pool(processes=num_proc) as pool:
                try:
                    ret = pool.starmap(launcher.run, iterable=params)
                except Exception as e:
                    logger.error('starmap: {}'.format(e))
                    sys.exit(1)
            logger.info(
                'Total duration: {}'.format(
                    timeit.default_timer() -
                    start_time))

            inputs_df = launcher.get_inputs_dataframe()
            for run_id, success, elapsed, elapsed_base in ret:
                inputs_df.at[run_id, CodeLauncher.SUCCESS_KEY] = success
                inputs_df.at[run_id, CodeLauncher.ELAPSED_TIME_KEY] = '{:.0f}'.format(
                    elapsed)
                inputs_df.at[run_id, CodeLauncher.ELAPSED_TIME_BASELINE_KEY] = '{:.0f}'.format(
                    elapsed_base)
            # Write the dataframe to the output directory
            launcher.write_dataframe(inputs_df, working_dir)

        elif launcher.load_inputs_dataframe():
            inputs_df = launcher.get_inputs_dataframe()
        else:
            logger.error(
                'Unable to load {}. Cannot continue'.format(
                    args.input))
            sys.exit(1)

        plot_spectra = options.get('plot_spectra', False)
        if plot_spectra:
            spectra = 'spectra.{}'.format(img_format)
            err = Plotter.aggregate_plots(
                output=spectra, working_dir=working_dir, legend=True)
            if err:
                logger.error('Error plotting aggregate spectrum')
                sys.exit(1)

        err, out_dict = Interpolator.interpolate_spectra(working_dir)
        interpolated_df = pd.DataFrame(out_dict)

        interpolated = paths.get('interpolated', None)
        if interpolated is not None:
            interpolated_file = os.path.join(working_dir, interpolated)
            if os.path.exists(interpolated_file):
                logger.warning(
                    '{} exists. Removing...'.format(interpolated_file))
                os.remove(interpolated_file)
            logger.info('Storing dict in file {}...'.format(interpolated_file))
            interpolated_df.to_csv(
                interpolated_file, index=False, na_rep='NaN')

        creator = DatasetCreator(
            inputs_df, interpolated_df, ('radius', 'slelints'))
        _, skipped = creator()
        logger.info('Skipped: {}'.format(skipped))

        dataset_file = os.path.join(
            working_dir, paths.get(
                'dataset', 'dataset.csv'))
        if os.path.exists(dataset_file):
            logger.warning(
                '{} exists and it will be overwritten.'.format(dataset_file))

        creator.write(dataset_file)

        logger.info('Done')
        sys.exit(0)
