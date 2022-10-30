#!/usr/bin/env python3
import common
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from tensorflow import keras


def de_normalize(data, min_val=-30, max_val=0):
    return min_val + (max_val - min_val) * data


def plot(y, i, label):
    plt.plot(y[i], '.', label=label)
    plt.ylim(-30, 0)
    # plt.xlim(300, 450)


model_path = sys.argv[1]

model = keras.models.load_model(model_path)

new_dataset = sys.argv[2]

new_set, new_test = common.load_data(new_dataset, 0) # returns train and test sets

y_d = de_normalize(new_set[1])

y_pred = model.predict(new_set[0])
y_pred_d = de_normalize(y_pred)

if len(sys.argv) > 3:
    out_dir = sys.argv[3]
else:
    out_dir = os.path.dirname(os.path.realpath(__file__))


plot(y_d, 0, 'actual')
plot(y_pred_d, 0, 'predicted')
plt.figure()

plot(y_d, 10, 'actual')
plot(y_pred_d, 10, 'predicted')

df_new = pd.DataFrame(np.transpose(y_d))
df_new.to_csv(os.path.join(out_dir, 'y.csv'), header=False, index=False)

df_pred = pd.DataFrame(np.transpose(y_pred_d))
df_pred.to_csv(os.path.join(out_dir, 'y_pred.csv'), header=False, index=False)

