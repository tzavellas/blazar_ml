#!/usr/bin/env python3
from astropy import constants as const
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.stats import ks_2samp
from tensorflow import keras


def calculate_error(y, y_pred, err_func):
    '''
    Calculates the error metric between two inputs using a given error function

    Parameters
    ----------
    y : ndarray
        The first input (actual).
    y_pred : ndarray
        The second input (prediction).
    err_func : function
        The error function (e.g. RMSE, RMSLE, etc.).

    Returns
    -------
    error_metric : ndarray
        The error metric.

    '''
    error_metric = np.zeros((len(y_pred), len(y)))
    for i, y_i in enumerate(y):
        for j, y_j in enumerate(y_pred):
            error_metric[j][i] = err_func(y_i, y_j[i])

    return error_metric


def calculate_predictions(models, input_set):
    '''
    Calculate the predictions of a given input set.

    Parameters
    ----------
    models : array
        Array of models.
    train_set : tuple(ndarray, ndarray)
        The input parameters and output values.

    Returns
    -------
    prediction : ndarray
        The predictions for each model.

    '''
    prediction = np.zeros((len(models), ) + input_set[1].shape) # preallocate
    for i, model in enumerate(models):
        prediction[i] = model.predict(input_set[0])    # index 0 means the case parameters

    return prediction


def compile_model(model, compile_kwargs={}, params={}):
    '''
    Compiles a model.

    Parameters
    ----------
    model : tf.keras.Model
        The model.
    compile_kwargs : dict, optional
        Contains optimizer, loss and metrics arguments for compile. The default is {}.
    params : dict, optional
        Contains the learnign rate of the optimizer. The default is {}.

    Returns
    -------
    model : tf.keras.Model
        The compiled model.

    '''
    learning_rate = params.get('learning_rate', 1e-3)

    optimizer = compile_kwargs.get('optimizer', keras.optimizers.Adam(learning_rate=learning_rate))
    loss=compile_kwargs.get('loss', keras.losses.MeanSquaredLogarithmicError())
    metrics=compile_kwargs.get('metrics', [keras.metrics.MeanSquaredLogarithmicError(),
                                           keras.metrics.MeanSquaredError()])

    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model


def de_normalize(data, min_val=-30, max_val=0):
    '''
    Denormalizes data so that the have values in range [min_val, max_val]

    Parameters
    ----------
    data : ndarray
        The normalized data.
    min_val : float, optional
        The minimul value. The default is -30.
    max_val : float, optional
        The maximum value. The default is 0.

    Returns
    -------
    ndarray
        The denormalized data.

    '''
    return min_val + (max_val - min_val) * data


def get_meta(architecture):
    '''
    Translates a dictionary to another. Key "inputs" converts to "n_features_in_"
    and key "outputs" converts to "n_outputs_expected_"

    Parameters
    ----------
    architecture : dict
        The input dictionary.

    Returns
    -------
    dict
        The translated dictionary.

    '''
    return {'n_features_in_': [architecture['inputs']],
            'n_outputs_expected_': architecture['outputs']}


def integral_error(y1, y2):
    """
    Wrapper function. Measures the integral under two curves and returns the
    square root of their squared difference.

    Parameters
    ----------
    y1 : ndarray
        Samples of the value of the first curve.
    y2 : ndarray
        Samples of the value of the second curve.

    Returns
    -------
    float
        The square root of the absolute difference of the areas.

    """
    area1 = simpson(y1)
    area2 = simpson(y2)
    return np.sqrt(np.square(area1 - area2))


def kolmogorov_smirnov_error(y1, y2):
    """
    Wrapper function. Performs a two-sample Kolmogorov-Smirnov test and returns
    the KS statistic as error.

    Parameters
    ----------
    y1 : ndarray
        Sample observations from the first continuous distribution.
    y2 : ndarray
        Sample observations from the second continuous distribution.

    Returns
    -------
    stat : float
        The error between the two distributions.

    """
    stat, p_value = ks_2samp(y1, y2)
    return stat


def load_data(path, test_ratio=0.2, random_state=42, drop_na=True, legacy=True, sample=True):
    '''
    Loads a dataset in csv format and returns a training set and a test set

    Parameters
    ----------
    path : string
        Path to csv dataset.
    test_ratio : float, optional
        Percentage of the dataset to use as test set. The default is 0.2.
    random_state : int, optional
        Seed for random number generation. The default is 42.
    drop_na : bool, optional
        Removes lines that contain NaN values. The default is True.
    legacy : bool, optional
        Ignores the last two columns. The default is True.
    sampe : bool, optional
        Reorders the dataset. The default is True.

    Returns
    -------
    Tuple of tuples of np.array
        The training set and the test set.

    '''
    raw_dataset = pd.read_csv(path, header=None, index_col=0, na_values='NaN')
    if drop_na:
        dataset = raw_dataset.dropna()
    else:
        dataset = raw_dataset

    if sample:
        train_set = dataset.sample(frac=1-test_ratio, random_state=random_state)
        test_set = dataset.drop(train_set.index)
    else:
        if test_ratio == 0:
            train_set = dataset
            test_set = pd.DataFrame()
        else:
            split_row = int(dataset.shape[0] * (1 - test_ratio))
            train_set = dataset[:split_row, :]
            test_set = dataset[split_row:, :]

    n_features = 6
    if legacy:
        x_train = train_set.iloc[:, :n_features].to_numpy()
        y_train = train_set.iloc[:, n_features:-1].to_numpy()

        x_test = test_set.iloc[:, :n_features].to_numpy()
        y_test = test_set.iloc[:, n_features:-1].to_numpy()
    else:
        x_train = train_set.iloc[:, :n_features].to_numpy()
        y_train = train_set.iloc[:, n_features:].to_numpy()

        x_test = test_set.iloc[:, :n_features].to_numpy()
        y_test = test_set.iloc[:, n_features:].to_numpy()

    return (x_train, y_train), (x_test, y_test)


def log_gamma_range(gamma, radius, bfield, 
                    base=10, 
                    q=const.e.esu.value,
                    m=const.m_e.cgs.value, 
                    c=const.c.cgs.value):
    '''
    Function that calculates the range of the geext values given a gamma value
    
    Parameters
    ----------
            gamma (float):          Reference logarithm gamma
            radius (float):         Reference logarithm radius
            bfield (float):         Reference logarithm bfield
            base (float):           The logarithm base, default value is 10
            q (float):              The electric charge, default value in cgs
            m (float):              The mass of the charge. default value in cgs
            c (float):              The speed of light, default value in cgs
    Returns
    -------
    Tuple
        The gamma geextmx range
    '''
    v_min = 2 + gamma
    logc = np.log(q / (m*c**2)) / np.log(base)
    v_max = radius + bfield + logc
    # set a hard upper limit based on code performance
    v_max = min(v_max, 8.)

    return v_min, v_max


def split_valid(x_train_full, y_train_full, ratio=0.2):
    '''
    Splits a full training set and returns a validation set and a training set

    Parameters
    ----------
    x_train_full : np.array
        The features of the training set.
    y_train_full : np.array
        The labels of the training set.
    ratio : float, optional
        Percentage of the full training set to use as validation set. The default is 0.2.

    Returns
    -------
    Tuple of tuples of np.array
        The validation set and the training set.

    '''
    n = int(len(x_train_full)*ratio)

    x_valid, x_train = x_train_full[:n], x_train_full[n:]
    y_valid, y_train = y_train_full[:n], y_train_full[n:]
    return (x_valid, y_valid), (x_train, y_train)