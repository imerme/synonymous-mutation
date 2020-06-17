import tensorflow as tf
from tensorflow import keras


def build_sub_model_1(**config):
    """
    A full connected neural networks
    :param config:dict, {‘ut_1’: 256, 'a': 'leak_relu', 'l1': 0., 'l2': 0., 'dp': 0.3, 'inputs_shape': (7,)}
    :return: model
    """
    if config['a'] == 'leaky_relu':
        config['a'] = tf.nn.leaky_relu
    inputs = keras.layers.Input(shape=config['inputs_shape'])

    # layer1
    x = keras.layers.Dense(config['ut_1'], activation=config['a'],
                           kernel_regularizer=keras.regularizers.l1_l2(config['l1'], config['l2']))(inputs)
    x = keras.layers.Dropout(config['dp'])(x)

    # layer2
    x = keras.layers.Dense(config['ut_2'], activation=config['a'],
                           kernel_regularizer=keras.regularizers.l1_l2(config['l1'], config['l2']))(x)
    x = keras.layers.Dropout(config['dp'])(x)

    # output
    out = keras.layers.Dense(2, activation='softmax')(x)

    # compile the model
    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(config['lr']), metrics=['accuracy'],
                  loss='categorical_crossentropy')
    return model


def build_sub_model_2(**config):
    """
    A convolutional neural networks
    :param config: dict: {'a': 'relu', 'inputs_shape': (919, 1), 'ft_1': 128, 'ft_2': 64, 'ps': 4, 'ks_1': 8, 'ks_2': 8}
    :return: model
    """
    if config['a'] == 'leaky_relu':
        config['a'] = tf.nn.leaky_relu
    inputs = keras.layers.Input(shape=config['inputs_shape'])

    # convolutional layer1
    x = keras.layers.Conv1D(filters=config['ft_1'], kernel_size=config['ks_1'], activation=config['a'],
                            padding='same')(inputs)
    x = keras.layers.MaxPooling1D(pool_size=config['ps'])(x)

    # convolutional layer2
    x = keras.layers.Conv1D(filters=config['ft_2'], kernel_size=config['ks_2'], activation=config['a'],
                            padding='same')(x)
    x = keras.layers.MaxPooling1D(pool_size=config['ps'])(x)

    # flatten layer and output
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(124, activation=config['a'])(x)
    out = keras.layers.Dense(2, activation='softmax')(x)

    # compile model
    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=keras.optimizers.Adam(config['lr']))
    return model


# def build_sub_model_3(**config):
#     # config = {'ft_1': 128, 'ks_1': 16, 'a': 'relu', 'inputs_shape': (919*2, 1)}
#     inputs = keras.layers.Input(shape=config['inputs_shape'])
#     x = keras.layers.Conv1D(config['ft_1'], kernel_size=config['ks_1'], strides=32, activation=config['a'])(inputs)
#     x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.3))(x)
#     x = keras.layers.Bidirectional(keras.layers.LSTM(64, dropout=0.3))(x)
#     x = keras.layers.Dropout(0.3)(x)
#     x = keras.layers.Dense(64, activation=config['a'])(x)
#     out = keras.layers.Dense(2, activation='softmax')(x)
#     # compile model
#     model = keras.Model(inputs=inputs, outputs=out)
#     model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=keras.optimizers.Adam(config['lr']))
#     return model


def merge_models(models, inputs, trainable_of_layers=True, **config):
    """
    A merge_models contain multi-channels, each channel receive a kind of data set such as tools scores and sequence
    features scores.
    :param models: a list of sub-models which are models trained by a kind of data set.
    :param inputs: a list of tuples which are shapes of the counterpart from models.
    :param trainable_of_layers: a bool to tell merge_model weather the params of sub-models if trainable.
    :param config: a dict to describe the merged part of the networks.like:
    {'dp': 0.3, 'ut_1': 128, 'ut_2': 64, 'a': 'relu'}
    :return: a model consist of multi-sub-models.
    """
    if config['a'] == 'leaky_relu':
        config['a'] = tf.nn.leaky_relu
    inputs = [keras.layers.Input(shape=shape) for shape in inputs]

    # establish sub-channels
    channels = len(models)
    outs = []
    for channel in range(channels):
        temp_x = inputs[channel]
        channel_model = models[channel]
        for layer in channel_model.layers[:-1]:
            layer.trainalbe = trainable_of_layers
            temp_x = layer(temp_x)
        outs.append(temp_x)

    # concatenate outputs from sub-channels and outputs
    merge_layer = keras.layers.Concatenate(axis=1)(outs)
    x = keras.layers.Dropout(config['dp'])(merge_layer)
    x = keras.layers.Dense(config['ut_1'], activation=config['a'])(x)
    x = keras.layers.Dropout(config['dp'])(x)
    x = keras.layers.Dense(config['ut_2'], activation=config['a'])(x)
    x = keras.layers.Dropout(config['dp'])(x)
    out = keras.layers.Dense(2, activation='softmax')(x)

    # compile models
    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(config['lr']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_merge_models(models, inputs, trainable_of_layers=True):
    def bm_(**config):
        return merge_models(models, inputs, trainable_of_layers=True, **config)
    return bm_



def build_sub_model_3(**config):
    """
    A convolutional neural networks
    :param config: dict: {'a': 'relu', 'inputs_shape': (919, 1), 'ft_1': 128, 'ft_2': 64, 'ps': 4, 'ks_1': 8, 'ks_2': 8}
    :return: model
    """
    if config['a'] == 'leaky_relu':
        config['a'] = tf.nn.leaky_relu

    # ref通道
    inputs_1 = keras.layers.Input(shape=config['inputs_shape'])
    # convolutional layer1
    x1 = keras.layers.Conv1D(filters=config['ft_1'], kernel_size=config['ks_1'], activation=config['a'],
                            padding='same')(inputs_1)
    x1 = keras.layers.MaxPooling1D(pool_size=config['ps'])(x1)

    # convolutional layer2
    x1= keras.layers.Conv1D(filters=config['ft_2'], kernel_size=config['ks_2'], activation=config['a'],
                            padding='same')(x1)
    x1 = keras.layers.MaxPooling1D(pool_size=config['ps'])(x1)

    # alt通道
    inputs_2 = keras.layers.Input(shape=config['inputs_shape'])
    # convolutional layer1
    x2 = keras.layers.Conv1D(filters=config['ft_1'], kernel_size=config['ks_1'], activation=config['a'],
                            padding='same')(inputs_2)
    x2 = keras.layers.MaxPooling1D(pool_size=config['ps'])(x2)

    # convolutional layer2
    x2= keras.layers.Conv1D(filters=config['ft_2'], kernel_size=config['ks_2'], activation=config['a'],
                            padding='same')(x2)
    x2 = keras.layers.MaxPooling1D(pool_size=config['ps'])(x2)

    # 合并通道ref 和 alt
    x = keras.layers.concatenate([x1, x2], axis=1)

    # flatten layer and output
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(124, activation=config['a'])(x)
    x = keras.layers.Dense(64, activation=config['a'])(x)
    out = keras.layers.Dense(2, activation='softmax')(x)

    # compile model
    model = keras.Model(inputs=[inputs_1, inputs_2], outputs=out)
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=keras.optimizers.Adam(config['lr']))
    return model


