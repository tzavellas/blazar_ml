#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 23:48:34 2022

@author: dino
"""

from astropy import constants as const
from astropy import units as u
import numpy as np
import os
import pandas as pd
import sys


def rmse(y_a, y_p, w = None):
    y1 = y_a.to_numpy()
    y2 = y_p.to_numpy()
    if w is not None:
        y1 = y1[w]
        y2 = y2[w]
    mse = np.square(np.subtract(y1, y2)).mean()
    return np.sqrt(mse)

def normalize_E(E):
    En = const.m_e * const.c**2
    En = En.to(u.eV)
    return np.log10(E/En.value)

if len(sys.argv) == 3:
    x_min = np.float64(sys.argv[1])
    x_max = np.float64(sys.argv[2])
else:
    x_min = 0.3e3 # eV
    x_max = 50e3  # eV

x_min = normalize_E(x_min)
x_max = normalize_E(x_max)


df_interpolation = pd.read_csv(sys.argv[1])
df_prediction = pd.read_csv(sys.argv[2], index_col=0)

df_interpolation = df_interpolation.dropna(axis=1)

rmses = np.zeros(df_prediction.shape[1])
s_rms = 0
count_rms = 0

x = df_interpolation['x'].to_numpy()
w = np.where(np.logical_and(x>= x_min, x<=x_max))

for name, y_predicted in df_prediction.iteritems():
    s = 'y_{}'.format(name) 
    if s in df_interpolation:
        y_actual = df_interpolation[s]
        rmse_i = rmse(y_actual, y_predicted, w)
        rmses[int(name)] = rmse_i
        s_rms = s_rms + rmse_i
        count_rms = count_rms + 1
    else:
        rmses[int(name)] = np.NaN

df = pd.DataFrame(rmses, columns=['RMSE'])
df.to_csv(sys.argv[3])
case = os.path.dirname(sys.argv[2]).split('/')[-1]
print('RMSE {}: {:.4f}'.format(case, s_rms / (count_rms - 1)))