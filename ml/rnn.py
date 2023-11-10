import tensorflow as tf


def build_simple_rnn(n_features, n_labels, n_hidden, n_neurons, name=None):
    input_shape = n_features, 1
    output_shape = n_labels

    model = tf.keras.models.Sequential(name=name)
    model.add(
        tf.keras.layers.SimpleRNN(
            n_neurons,
            input_shape=input_shape,
            return_sequences=True,
            dtype=tf.float64))
    for layer in range(n_hidden - 1):
        model.add(tf.keras.layers.SimpleRNN(n_neurons, return_sequences=True))
    model.add(tf.keras.layers.SimpleRNN(n_neurons))
    model.add(
        tf.keras.layers.Dense(
            output_shape,
            activation='softplus'))

    return model


def build_lstm(n_features, n_labels, n_hidden, n_neurons, name=None):
    input_shape = n_features, 1
    output_shape = n_labels

    model = tf.keras.models.Sequential(name=name)
    model.add(
        tf.keras.layers.LSTM(
            n_neurons,
            input_shape=input_shape,
            return_sequences=True,
            dtype=tf.float64))
    for layer in range(n_hidden):
        model.add(tf.keras.layers.LSTM(n_neurons, return_sequences=True))
    model.add(tf.keras.layers.LSTM(n_neurons))
    model.add(
        tf.keras.layers.Dense(
            output_shape,
            activation='softplus'
            ))

    return model


def build_gru(n_features, n_labels, n_hidden, n_neurons, name=None):
    input_shape = n_features, 1
    output_shape = n_labels

    model = tf.keras.models.Sequential(name=name)
    model.add(
        tf.keras.layers.GRU(
            n_neurons,
            input_shape=input_shape,
            return_sequences=True,
            dtype=tf.float64))
    for layer in range(n_hidden):
        model.add(tf.keras.layers.GRU(n_neurons, return_sequences=True))
    model.add(tf.keras.layers.GRU(n_neurons))
    model.add(
        tf.keras.layers.Dense(
            output_shape,
            activation='softplus'
            ))

    return model
