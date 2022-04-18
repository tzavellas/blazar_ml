#!/usr/bin/python


import numpy as np
import pandas as pd
import argparse
from astropy import constants as const



class sample():
    def __init__(self, low, high, f_sample=np.random.uniform):
        self.low        = low
        self.high       = high
        self.f_sample   = f_sample
    
    def __call__(self):
        return self.f_sample(self.low, self.high)
    


def log_gamma_range(loggamma, logR, logB, 
                    base=10, 
                    q=const.e.esu.value,
                    m=const.m_e.cgs.value, 
                    c=const.c.cgs.value):
    '''
    Function that calculates the range of the log_gamma_max values given the a
    loggamma value

            Parameters:
                    loggamma (float): Reference loggamma
                    logR (float): Reference logR
                    logB (float): Reference logB
                    base (float): The logarithm base, default value is 10
                    q (float): The electric charge, default value in cgs
                    m (float): The mass of the charge. default value in cgs
                    c (float): The speed of light, default value in cgs
    '''
    v_min = 2 + loggamma
    logc = np.log(q / (m*c**2)) / np.log(base)
    v_max = logR + logB + logc
    # set a hard upper limit based on code performance
    if v_max > 8.: 
        v_max = 8.
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
    limits = {'log_R'       : sample(14, 17),
              'log_B'       : sample(-2, 2),
              'log_gamma'   : sample(0.1, 4),
              'log_le'      : sample(-5, -1),
              'p'           : sample(1.5, 4)
              }

    # Pre-allocate arrays of samples
    log_R           = np.zeros(N)
    log_B           = np.zeros(N)
    log_gamma_min   = np.zeros(N)
    log_gamma_max   = np.zeros(N)
    log_le          = np.zeros(N)
    p               = np.zeros(N)
    
    for i in range(N):
        # Draw samples from each distribution
        log_R[i]         = limits['log_R']()
        log_B[i]         = limits['log_B']()
        log_gamma_min[i] = limits['log_gamma']()
        log_le[i]        = limits['log_le']()
        p[i]             = limits['p']()
        
        # Special handling - depends on gamma_min, R and B
        log_gamma_max[i] = sample(*log_gamma_range(log_gamma_min[i],
                                                   log_R[i],
                                                   log_B[i]))()
    
    
    # Store arrays in a dataframe
    df = pd.DataFrame()
    df['log_R'] = log_R
    df['log_B'] = log_B
    df['log_gamma_min'] = log_gamma_min
    df['log_gamma_max'] = log_gamma_max
    df['log_le'] = log_le
    df['p'] = p
    
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
