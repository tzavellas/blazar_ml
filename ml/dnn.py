import tensorflow as tf


def base_dense(input_shape, output_shape, hidden_layer_sizes, name):
    '''
    Creates a model of a dense network, given a set of parameters.

    Parameters
    ----------
    input_shape : list[int]
        The input dimensions.
    output_shape : list[int]
        The output dimensions.
    hidden_layer_sizes : list[int]
        List of integers, representing the number of neurons, of each hidden layer.

    Returns
    -------
    model : tf.keras.Model
        The model.

    '''
    input_layer = tf.keras.layers.Input(shape=input_shape)
    hidden = input_layer
    for n_neurons in hidden_layer_sizes:
        hidden = tf.keras.layers.Dense(n_neurons, activation="relu")(hidden)
    output_layer = tf.keras.layers.Dense(output_shape)(hidden)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name=name)
    return model


def build_model(features, labels, n_hidden, n_neurons, name=None):
    input_shape = features
    output_shape = labels

    hidden_layer_sizes = [n_neurons for i in range(n_hidden)]
    model = base_dense(input_shape, output_shape, hidden_layer_sizes, name=name)

    return model


def build_model_avg(features, labels, n_hidden, n_neurons, n_base=2, name=None):
    input_shape = features
    output_shape = labels

    hidden_layer_sizes = [n_neurons for i in range(n_hidden)]

    input_layer = tf.keras.layers.Input(shape=input_shape)
    base_out = base_dense(input_shape, output_shape, hidden_layer_sizes, name)
    layers = [base_out(input_layer) for i in range(n_base)]
    avg = tf.keras.layers.Average()(layers)
    output_layer = tf.keras.layers.Dense(output_shape)(avg)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name=name)

    return model


def build_model_concat(features, labels, n_hidden, n_neurons, n_base=2, name=None):
    input_shape = features
    output_shape = labels

    hidden_layer_sizes = [n_neurons for i in range(n_hidden)]

    input_layer = tf.keras.layers.Input(shape=input_shape)
    base_out = base_dense(input_shape, output_shape, hidden_layer_sizes, name)
    layers = [base_out(input_layer) for i in range(n_base)]
    merged = tf.keras.layers.concatenate(layers)
    output_layer = tf.keras.layers.Dense(
        output_shape, activation='relu')(merged)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name=name)

    return model
