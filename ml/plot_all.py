#!/usr/bin/env python3
import common
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tensorflow import keras
from sklearn import metrics


def avg_error_ranking(error, model_ids):
    m_id = np.array(model_ids)
    mm = np.mean(error, axis=1)
    ss = np.std(error, axis=1)
    argsort = np.argsort(mm)    
    return m_id[argsort].tolist(), list(zip(mm,ss))


def mean_ranking(error, model_ids):
    argsort = np.argsort(error, axis=0)
    rankings = argsort + 1
    mean = np.mean(rankings, axis=1)
    m_id = np.array(model_ids)
    argsort = np.argsort(mean)
    m_id[argsort].tolist()
    return m_id[argsort].tolist(), mean, rankings
    

def de_normalize(data, min_val=-30, max_val=0):
    return min_val + (max_val - min_val) * data


def plot(y, marker, label):
    plt.plot(y, marker=marker, markersize=1, linestyle='', label=label)
    plt.ylim(-30, 0)
    # plt.xlim(300, 450)


def plot_matrix(data, name, labels, symbols='ox+|_'):
    fig = plt.figure()
    for i, y in enumerate(data):
        plt.plot(y, marker=symbols[i], linestyle='',  label=labels[i])
        plt.xlabel('case')
        plt.ylabel(name)
        plt.legend()
    fig_path = os.path.join(out_dir, '{}.svg'.format(name))
    plt.savefig(fig_path, format='svg')
    plt.close(fig)

    
def plot_all(y, prediction, labels, symbols='ox+|_'):
    y_pred = de_normalize(prediction)
    y_d = de_normalize(y)    # index 1 means the spectrum values
    for i, y_i in enumerate(y_d):
        fig = plt.figure(i)
        plot(y_i, symbols[0], 'actual')
        for j, y_p in enumerate(y_pred):
            plot(y_p[i], symbols[j + 1], labels[j])
        plt.legend()
        fig_path = os.path.join(out_dir, '{}.svg'.format(i))
        plt.savefig(fig_path, format='svg')
        plt.close(fig)
    
    
def calculate_error(y, y_pred, err_func):
    error_metric = np.zeros((len(y_pred), len(y)))
    for i, y_i in enumerate(y):
        for j, y_j in enumerate(y_pred):
            error_metric[j][i] = err_func(y_i, y_j[i])

    return error_metric


if __name__ == "__main__":

    new_dataset = sys.argv[1]   # load dataset
    train_set, test_set = common.load_data(new_dataset, 0) # returns train and test sets

    if len(sys.argv) > 2:       # Read/create output dir
        out_dir = sys.argv[2]
    else:
        out_dir = 'plots'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    model_files = ['hea_dnn.h5', 'hea_rnn.h5', 'hea_gru.h5', 'hea_lstm.h5' ]
    models = [keras.models.load_model(m) for m in model_files]

    prediction = np.zeros((len(model_files), ) + train_set[1].shape) # preallocate
    for i, model in enumerate(models):
        prediction[i] = model.predict(train_set[0])    # index 0 means the case parameters

    error_metric = calculate_error(train_set[1], prediction, metrics.mean_squared_error)
    
    rank_avg, stats = avg_error_ranking(error_metric, model_files)
    
    avg_rank, mean_rank, rankings = mean_ranking(error_metric, model_files)
        
    plot_matrix(error_metric, 'error', model_files)
    plot_matrix(rankings, 'ranking', model_files)
    plot_all(train_set[1], prediction, model_files)