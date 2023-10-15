import tensorflow as tf


def base_dense(input_shape, output_shape, neurons_p_layer, name):
    '''
    Creates a model of a dense network, given a set of parameters.

    Parameters
    ----------
    input_shape : list[int]
        The input dimensions.
    output_shape : list[int]
        The output dimensions.
    neurons_p_layer : list[int]
        List of integers, representing the number of neurons, of each hidden layer.

    Returns
    -------
    model : tf.keras.Model
        The model.

    '''
    input_layer = tf.keras.layers.Input(shape=input_shape)
    hidden = input_layer
    for n_neurons in neurons_p_layer:
        hidden = tf.keras.layers.Dense(n_neurons, activation="relu")(hidden)
    output_layer = tf.keras.layers.Dense(output_shape,
                                         activation=tf.keras.layers.LeakyReLU(0.2))(hidden)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name=name)
    return model


def build_model(n_features, n_labels, n_hidden, n_neurons, name=None):
    neurons_p_layer = [n_neurons for i in range(n_hidden)]
    model = base_dense(n_features, n_labels, neurons_p_layer, name=name)

    return model


def build_model_avg(n_features, n_labels, n_hidden, n_neurons, n_base=2, name=None):
    neurons_p_layer = [n_neurons for i in range(n_hidden)]

    input_layer = tf.keras.layers.Input(shape=input_shape)
    base_out = base_dense(n_features, n_labels, neurons_p_layer, name)
    layers = [base_out(input_layer) for i in range(n_base)]
    avg = tf.keras.layers.Average()(layers)
    output_layer = tf.keras.layers.Dense(n_labels)(avg)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name=name)

    return model


def build_model_concat(n_features, n_labels, n_hidden, n_neurons, n_base=2, name=None):
    input_shape = n_features
    output_shape = n_labels

    neurons_p_layer = [n_neurons for i in range(n_hidden)]

    input_layer = tf.keras.layers.Input(shape=n_features)
    base_out = base_dense(input_shape, n_labels, neurons_p_layer, name)
    layers = [base_out(input_layer) for i in range(n_base)]
    merged = tf.keras.layers.concatenate(layers)
    output_layer = tf.keras.layers.Dense(n_labels,
                                         activation=tf.keras.layers.LeakyReLU(0.2))(merged)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name=name)

    return model
