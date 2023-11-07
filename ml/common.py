from astropy import constants as const
import numpy as np
import os
import pandas as pd
import keras_tuner as kt
from scipy import integrate, stats
import tensorflow as tf


def calculate_error(y, y_pred, err_func):
    """
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

    """
    m = y_pred.shape[0]
    n = y.shape[0]
    error_metric = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            error_metric[i, j] = err_func(y[j, :], y_pred[i, j, :])
    return error_metric


def calculate_predictions(models, input_set):
    """
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

    """
    prediction = np.zeros((len(models), ) + input_set[1].shape)  # preallocate
    for i, model in enumerate(models):
        # index 0 means the case parameters
        prediction[i] = model.predict(input_set[0])

    return prediction


def de_normalize(data, min_val=-30, max_val=0):
    """
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

    """
    return min_val + (max_val - min_val) * data


def de_normalize2(y, min_val=-30, max_val=0):
    m, n = y.shape
    y_d = np.zeros((m, n - 1))
    for i in range(m):
        max_el = y[i, -1] * min_val
        y_d[i] = y[i, :-1] * (np.abs(min_val) + max_el) - np.abs(min_val)

    return y_d


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
    area1 = integrate.simpson(y1)
    area2 = integrate.simpson(y2)
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
    stat, p_value = stats.ks_2samp(y1, y2)
    return stat


def scheduler(epoch, lr, rate=0.1):
    """
    Constant exponential decay schedule function with configurable rate. Keeps
    the initial learning rate for the first 10 epochs.

    Parameters
    ----------
    epoch : int
        The current training epoch.
    lr : float
        The current learning rate.
    rate : float, optional
        The decay rate. Default value is 0.1.

    Returns
    -------
    float
        Learning rate reduced by a factor of exp(-rate).

    """
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-rate)


def load_data(path, n_features, test_ratio=0.2,
              random_state=42, drop_na=True, sample=True):
    """
    Loads a dataset in csv format and returns a training set and a test set

    Parameters
    ----------
    path : string
        Path to csv dataset.
    n_features : int
        Number of features.
    test_ratio : float, optional
        Percentage of the dataset to use as test set. The default is 0.2.
    random_state : int, optional
        Seed for random number generation. The default is 42.
    drop_na : bool, optional
        Removes lines that contain NaN values. The default is True.
    sample : bool, optional
        Reorders the dataset. The default is True.

    Returns
    -------
    Tuple of lists of np.array
        The training set and the test set.

    """
    raw_dataset = pd.read_csv(path, header=None, na_values='NaN')
    if drop_na:
        dataset = raw_dataset.dropna()
    else:
        dataset = raw_dataset

    if sample:
        train_set = dataset.sample(
            frac=1 - test_ratio,
            random_state=random_state)
        test_set = dataset.drop(train_set.index)
    else:
        if test_ratio == 0:
            train_set = dataset
            test_set = pd.DataFrame()
        else:
            split_row = int(dataset.shape[0] * (1 - test_ratio))
            train_set = dataset[:split_row, :]
            test_set = dataset[split_row:, :]

    # x -> slice from 0 to n_features
    # y -> slice from n_features to the end
    x_train = train_set.iloc[:, :n_features].to_numpy()
    y_train = train_set.iloc[:, n_features:].to_numpy()

    x_test = test_set.iloc[:, :n_features].to_numpy()
    y_test = test_set.iloc[:, n_features:].to_numpy()

    return [x_train, y_train], [x_test, y_test]


def log_gamma_range(gamma, radius, bfield,
                    base=10,
                    q=const.e.esu.value,
                    m=const.m_e.cgs.value,
                    c=const.c.cgs.value):
    """
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
    """
    if hasattr(gamma, '__len__'):
        v_min = 2 + gamma
        logc = np.ones(gamma.shape) * np.log(q / (m * c**2)) / np.log(base)
        v_max = radius + bfield + logc
        v_max = np.minimum(v_max, np.ones(gamma.shape) * 8)

        return v_min, v_max
    else:
        v_min = 2 + gamma
        logc = np.log(q / (m * c**2)) / np.log(base)
        v_max = radius + bfield + logc
        # set a hard upper limit based on code performance
        v_max = min(v_max, 8.)

        return v_min, v_max


def normalize_clip(data, min_val=-30, max_val=0):
    """
    Clips and normalizes data so that the have values in range [0, 1]

    Parameters
    ----------
    data : ndarray
        The data.
    min_val : float, optional
        The minimul value. The default is -30.
    max_val : float, optional
        The maximum value. The default is 0.

    Returns
    -------
    ndarray
        The normalized data.

    """
    clipped = np.clip(data, min_val, max_val)
    return (clipped - min_val) / (max_val - min_val)


def normalize_clip2(y, min_val=-30, max_val=0):
    clipped = np.clip(y, min_val, max_val)
    m, n = clipped.shape
    y_n = np.zeros((m, n + 1))
    for i in range(m):
        max_el = np.amax(clipped[i])
        yn = (clipped[i] + np.abs(min_val)) / (np.abs(min_val) + max_el)
        y_n[i] = np.append(yn, max_el / min_val)
    return y_n


def split_valid(x_train_full, y_train_full, ratio=0.2):
    """
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

    """
    n = int(len(x_train_full) * ratio)

    x_valid, x_train = x_train_full[:n], x_train_full[n:]
    y_valid, y_train = y_train_full[:n], y_train_full[n:]
    return (x_valid, y_valid), (x_train, y_train)


def check_environment(config):
    """Updates the config path variable, if it has the special
    value ENV_VARIABLE_PLACEHOLDER. If it does, it uses the
    resolved 'HEA_DATASET_PATH' environment variable as path.

    Args:
        config (dict): Dictionary representation of the config file.

    Raises:
        ValueError: if path equals 'ENV_VARIABLE_PLACEHOLDER'
        and 'HEA_DATASET_PATH' is not set or the resolved
        environment variable does not exist.

    Returns:
        dict: The updated config
    """
    if config['dataset']['path'] == 'ENV_VARIABLE_PLACEHOLDER':
        env_var_value = os.getenv('HEA_DATASET_PATH')
        if env_var_value:
            if os.path.exists(env_var_value):
                config['dataset']['path'] = env_var_value
            else:
                raise ValueError(
                    f'HEA_DATASET_PATH={env_var_value} does not exist')
        else:
            raise ValueError(
                f'Environment variable HEA_DATASET_PATH is not set!')
    return config


class Tuner(kt.HyperModel):

    def __init__(self, build_func, n_features, n_labels, hyper_params):
        self.build_func = build_func
        self.features = n_features
        self.labels = n_labels
        self.neuron = hyper_params['neuron']
        self.hidden = hyper_params['hidden']
        self.lr = hyper_params['learning_rate']
        self.batch = hyper_params['batch_size']

    def build(self, hp):
        # Number of neurons is a hyper parameter
        neurons = hp.Int('neurons',
                         min_value=self.neuron[0],
                         max_value=self.neuron[1])
        # Number of hidden layers is a hyper parameter
        hidden = hp.Int('hidden',
                        min_value=self.hidden[0],
                        max_value=self.hidden[1] - 1,
                        step=1)
        # Learning rate is a hyper parameter
        lr = hp.Float('learning_rate',
                      min_value=self.lr[0],
                      max_value=self.lr[1])
        # Batch size is a hyper parameter
        hp.Choice('batch_size', values=self.batch)

        model = self.build_func(self.features, self.labels, hidden, neurons)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.MeanSquaredLogarithmicError(),
                      metrics=tf.keras.metrics.MeanSquaredError())
        return model
