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
                                         activation='softplus')(hidden)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name=name)
    return model


def build_model(n_features, n_labels, n_hidden, n_neurons, name=None):
    neurons_p_layer = [n_neurons for i in range(n_hidden)]
    model = base_dense(n_features, n_labels, neurons_p_layer, name=name)

    return model
