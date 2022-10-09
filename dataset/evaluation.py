#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 23:48:34 2022

@author: dino
"""

import numpy as np
import pandas as pd
import sys


def rmse(y_a, y_p):
    mse = np.square(np.subtract(y_a, y_p)).mean()
    return np.sqrt(mse)


df_interpolation = pd.read_csv(sys.argv[1])
df_prediction = pd.read_csv(sys.argv[2], index_col=0)

df_interpolation = df_interpolation.dropna(axis=1)

rmses = np.zeros(df_prediction.shape[1])
s_rms = 0
count_rms = 0

for name, y_predicted in df_prediction.iteritems():
    s = 'y_{}'.format(name) 
    if s in df_interpolation:
        y_actual = df_interpolation[s]
        rmse_i = rmse(y_actual, y_predicted)
        rmses[int(name)] = rmse_i
        s_rms = s_rms + rmse_i
        count_rms = count_rms + 1
    else:
        rmses[int(name)] = np.NaN

df = pd.DataFrame(rmses, columns=['RMSE'])
df.to_csv(sys.argv[3])
print('Average RMSE: {}'.format(s_rms / (count_rms - 1)))