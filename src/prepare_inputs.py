#!/usr/bin/python


import numpy as np
import pandas as pd
import argparse
from astropy import constants as const


RADIUS      = 'radius'
BFIELD      = 'bfield'
GAMMA       = 'geext'
GAMMA_MIN   = 'geextmn'
GAMMA_MAX   = 'geextmx'
LE          = 'exlumel'
P           = 'belesc'

class sample():
    def __init__(self, low, high, f_sample=np.random.uniform):
        self.low        = low
        self.high       = high
        self.f_sample   = f_sample
    
    def __call__(self):
        return self.f_sample(self.low, self.high)
    


def log_gamma_range(gamma, radius, bfield, 
                    base=10, 
                    q=const.e.esu.value,
                    m=const.m_e.cgs.value, 
                    c=const.c.cgs.value):
    '''
    Function that calculates the range of the geext values given the a gamma value
            Parameters:
                    gamma (float): Reference logarithm gamma
                    radius (float): Reference logarithm radius
                    bfield (float): Reference logarithm bfield
                    base (float): The logarithm base, default value is 10
                    q (float): The electric charge, default value in cgs
                    m (float): The mass of the charge. default value in cgs
                    c (float): The speed of light, default value in cgs
    '''
    v_min = 2 + gamma
    logc = np.log(q / (m*c**2)) / np.log(base)
    v_max = radius + bfield + logc
    # set a hard upper limit based on code performance
    v_max = min(v_max, 8.)

    return v_min, v_max


def generate_sample_inputs(N, out):
    '''
    Creates a csv file with a number of input parameters for the SSC model 
    execution.

            Parameters:
                    N (int): Number of inputs
                    out (str): The name of the output file
    '''
    
    # Dictionary of sampling callables
    limits = {RADIUS    : sample(14, 17),
              BFIELD    : sample(-2, 2),
              GAMMA     : sample(0.1, 4),
              LE        : sample(-5, -1),
              P         : sample(1.5, 4)
              }

    # Pre-allocate arrays of samples
    radius          = np.zeros(N)
    bfield          = np.zeros(N)
    geextmn         = np.zeros(N)
    geextmx         = np.zeros(N)
    exlumel         = np.zeros(N)
    belesc          = np.zeros(N)
    
    for i in range(N):
        # Draw samples from each distribution
        radius[i]       = limits[RADIUS]()
        bfield[i]       = limits[BFIELD]()
        geextmn[i]      = limits[GAMMA]()
        exlumel[i]      = limits[LE]()
        belesc[i]       = limits[P]()
        
        # Special handling - depends on gamma_min, R and B
        geextmx[i] = sample(*log_gamma_range(geextmn[i], radius[i], bfield[i]))()
    
    
    # Store arrays in a dataframe
    df = pd.DataFrame()
    df[RADIUS]      = radius
    df[BFIELD]      = bfield
    df[GAMMA_MIN]   = geextmn
    df[GAMMA_MAX]   = geextmx
    df[LE]          = exlumel
    df[P]           = belesc
    
    # write to file
    df.to_csv(out, index=True, index_label='run',
              float_format='%.6e', line_terminator='\n')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates a sample of the input space')
    parser.add_argument('--size', default=10, type=int, help='Size of the sample')
    parser.add_argument('--out', default='out.csv', type=str, help='Output file')
    
    try:
        args = parser.parse_args()
        generate_sample_inputs(args.size, args.out)
    
    except argparse.ArgumentError:
        print('Error parsing arguments')
