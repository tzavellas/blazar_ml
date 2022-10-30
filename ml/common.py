#!/usr/bin/env python3
import pandas as pd


def load_data(path, test_ratio=0.2, random_state=42, drop_na=True):
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

    train_set = dataset.sample(frac=1-test_ratio, random_state=random_state)
    test_set = dataset.drop(train_set.index)

    n_features = 6

    x_train = train_set.iloc[:, :n_features].to_numpy()
    y_train = train_set.iloc[:, n_features:-1].to_numpy()

    x_test = test_set.iloc[:, :n_features].to_numpy()
    y_test = test_set.iloc[:, n_features:-1].to_numpy()

    return (x_train, y_train), (x_test, y_test)


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