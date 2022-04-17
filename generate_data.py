#!/usr/bin/python


import numpy as np
import pandas as pd
import argparse


class sample():
    def __init__(self, low, high, f_sample=np.random.uniform):
        self.low        = low
        self.high       = high
        self.f_sample   = f_sample
    
    def __call__(self):
        return self.f_sample(self.low, self.high)
    

def gamma_min(gamma, multi=1e2):
    '''
    Returns the value of gamma times a multiplier.

            Parameters:
                    gamma (float): Gamma
                    multi (float): The multiplier
    '''
    return 1e2 * gamma


def gamma_max(R, B, q=4.8032e-10, m=9.1094e-28, c=3e10):
    '''
    Returns the value of gamma that results from the gyro-radius formula.
    Default values are in cgs units

            Parameters:
                    R (float): The gyro-radius
                    B (float): The magnetic field
                    q (float): The electric charge
                    m (float): The mass of the charge
                    c (float): The speed of light
    '''
    return R * B * q / (m * c)


def gamma_(log_gamma, log_R, log_B, base=10):
    '''
    Wrapper function that converts from the reference log_base values and 
    returns a tuple of (min max) values

            Parameters:
                    log_gamma (float): Reference log_gamma
                    log_R (float): Reference log_R
                    log_B (float): Reference log_B
                    base (float): The log base, default value is 10
    '''
    g = np.power(base, log_gamma)
    R = np.power(base, log_R)
    B = np.power(base, log_B)
    v_min = gamma_min(g)
    v_max = gamma_max(R, B)
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
              'log_B'       : sample(-3, 2),
              'log_gamma'   : sample(0, 4),
              'log_le'      : sample(-5, -1),
              'p'           : sample(1, 4)
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
        log_gamma_max[i] = sample(*gamma_(log_gamma_min[i], 
                                         log_R[i],
                                         log_B[i])
                                  )()
    
    
    # Store arrays in a dataframe
    df = pd.DataFrame()
    df['log_R'] = log_R
    df['log_B'] = log_B
    df['log_gamma_min'] = log_gamma_min
    df['log_gamma_max'] = log_gamma_max
    df['log_le'] = log_le
    df['p'] = p
    
    # write to file
    df.to_csv(out, index=True, float_format='%.6e', line_terminator='\n')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates a sample of the input space')
    parser.add_argument('--size', default=10, type=int, help='Size of the sample')
    parser.add_argument('--out', default='out.csv', type=str, help='Output file')
    
    try:
        args = parser.parse_args()
        generate_sample_inputs(args.size, args.out)
    
    except argparse.ArgumentError:
        print('Error parsing arguments')