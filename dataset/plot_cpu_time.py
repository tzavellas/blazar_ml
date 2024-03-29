import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Generates scatter plots of CPU time vs parameters.')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Full path to parameter csv.')
    parser.add_argument('-w', '--working-dir', type=str, default='output',
                        help='Root path where the dataset will be stored. Default is "output".')
    parser.add_argument('-f', '--format', type=str, default='png',
                        help='Spectrum image format. Default is png.')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        return 1

    file = args.input

    if not os.path.exists(file):
        print('File {} does not exist'.format(file))
        return 1

    df = pd.read_csv(file)

    y = np.array(df['elapsed_time'])
    nz = np.where(y > 1)  # keep only non zero entries
    yf = y[nz]

    keys = ['radius', 'bfield', 'exlumel']  # keys to plot
#   ADDED extra diagnostic computed from the 3 keys above
    bfield_comp = 8.125e-19 * \
        (10**np.array(df['bfield']))**2/(8*np.pi)*10**np.array(df['radius'])
    lelb = 10.**np.array(df['exlumel'])/bfield_comp

    print('Plots wiil be saved at {}'.format(args.working_dir))

    for key in keys:
        fig = plt.figure(key)
        x = np.array(df[key])
        xf = x[nz]
        plt.plot(xf, yf, '.', figure=fig)
        plt.xlabel(key)
        plt.ylabel('CPU time (s)')
        plt.ylim((0, 210))
        fout = os.path.join(args.working_dir, '{}.{}'.format(key, args.format))
        plt.savefig(fout)
        plt.close(fig)

# Additional diagnostic plot
    fig = plt.figure()
    x = lelb
    xf = x[nz]
    plt.plot(np.log10(xf), yf, '.', figure=fig)
    plt.xlabel('log(le/lb)')
    plt.ylabel('CPU time (s)')
    plt.ylim((0, 210))
    fout = os.path.join(args.working_dir, '{}.{}'.format('lelb', args.format))
    plt.savefig(fout)
    plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
