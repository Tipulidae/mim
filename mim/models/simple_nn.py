import math
from copy import deepcopy

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv1D,
    MaxPool1D,
    Dropout,
    BatchNormalization,
    Concatenate,
    ReLU,
    AveragePooling1D
)

from mim.models.load import load_model_from_experiment_result
from mim.util.logs import get_logger

log = get_logger('simple_nn')


def super_basic_cnn(train, validation=None, dropout=0, filters=32,
                    kernel_size=16, pool_size=8, hidden_size=10):
    inp = {key: Input(shape=value) for key, value in train['x'].shape.items()}
    x = inp['ecg']
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        kernel_regularizer="l2",
        padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool1D(pool_size=pool_size)(x)
    x = Dropout(dropout)(x)

    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        kernel_regularizer="l2",
        padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool1D(pool_size=pool_size)(x)
    x = Dropout(dropout)(x)

    x = Flatten()(x)
    x = Dense(hidden_size, activation="relu")(x)
    x = Dropout(dropout)(x)

    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)


def basic_cnn2(train, validation=None, dropout=0, layers=None,
               hidden_layer=None, batch_norm=True):
    inp = {key: Input(shape=value) for key, value in train['x'].shape.items()}
    x = inp['ecg']
    for layer in layers:
        x = Conv1D(
            filters=layer['filters'],
            kernel_size=layer['kernel_size'],
            kernel_regularizer="l2",
            padding='same')(x)
        if batch_norm:
            x = BatchNormalization()(x)

        x = ReLU()(x)
        x = MaxPool1D(pool_size=2)(x)
        x = Dropout(dropout)(x)

    x = Flatten()(x)
    if hidden_layer:
        x = Dense(10)(x)
        x = Dropout(hidden_layer['dropout'])(x)

    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)


def basic_cnn3(train, validation=None, dropout=0, layers=None,
               pool_size=2, hidden_dropout=None):
    inp = {key: Input(shape=value) for key, value in train['x'].shape.items()}
    x = BatchNormalization()(inp['ecg'])

    n = len(layers)
    max_pool_size = math.floor((train['x']['ecg'].shape[0] / 20) ** (1 / n))
    pool_size = min(max_pool_size, pool_size)
    for layer in layers:
        x = Conv1D(
            filters=layer['filters'],
            kernel_size=layer['kernel_size'],
            kernel_regularizer="l2",
            padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool1D(pool_size=pool_size)(x)
        x = Dropout(layer['dropout'])(x)

    x = Flatten()(x)
    if hidden_dropout is not None:
        x = Dense(10)(x)
        x = Dropout(hidden_dropout)(x)

    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)


def ecg_cnn(train, validation=None, dense_size=10, dropout=0.3,
            cnn_kwargs=None):
    inp = {key: Input(shape=value) for key, value in train['x'].shape.items()}
    layers = []
    if 'ecg_0' in inp:
        layers.append(ecg_network2(inp['ecg_0'], **cnn_kwargs))
    if 'ecg_1' in inp:
        layers.append(ecg_network2(inp['ecg_1'], **cnn_kwargs))
    if 'flat_features' in inp:
        layers.append(BatchNormalization()(inp['flat_features']))

    if len(layers) > 1:
        x = Concatenate()(layers)
    else:
        x = layers[0]

    x = Dense(dense_size, activation='relu')(x)
    x = Dropout(dropout)(x)
    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)


def _ecg_network(x, num_conv_layers, dropout=0.3, filters=32,
                 kernel_size=16, output_size=10, pool_size=16):
    for _ in range(num_conv_layers):
        x = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            kernel_regularizer="l2",
            padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool1D(pool_size=pool_size)(x)
        x = Dropout(dropout)(x)

    x = Flatten()(x)
    x = Dense(output_size, activation="relu")(x)
    return Dropout(dropout)(x)


def ecg_network2(x, num_layers=2, dropout=0.3, filter_first=16,
                 filter_last=16, kernel_first=5, kernel_last=5,
                 dense=True, batch_norm=True, pool_size=None,
                 downsample=False, dense_size=10):

    if downsample:
        x = AveragePooling1D(2, padding='same')(x)
    if pool_size is None:
        pool_size = _calculate_appropriate_pool_size(
            input_size=x.shape[1],
            num_pools=num_layers,
            minimum_output_size=4
        )

    filters = map(round, np.linspace(filter_first, filter_last, num_layers))
    kernels = map(round, np.linspace(kernel_first, kernel_last, num_layers))
    for filter_size, kernel_size in zip(filters, kernels):
        x = Conv1D(
            filters=filter_size,
            kernel_size=kernel_size,
            kernel_regularizer="l2",
            padding='same')(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool1D(pool_size=pool_size)(x)
        x = Dropout(dropout)(x)

    x = Flatten()(x)
    if dense:
        x = Dense(dense_size, activation="relu")(x)
        x = Dropout(dropout)(x)
    return x


def _calculate_appropriate_pool_size(
        input_size, num_pools, minimum_output_size=4):
    """
    Calculate what the pool size should be if we start with input_size and
    pool num_pools times, and want to end up with a size that is at least
    minimum_output_size.
    :param input_size:
    :param num_pools:
    :param minimum_output_size:
    :return:
    """
    return math.floor((input_size / minimum_output_size) ** (1 / num_pools))


def basic_ff():
    inp = Input(shape=(128, ))
    x = Flatten()(inp)
    x = Dense(32, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)
    model = keras.Model(inp, output)
    return model


def serial_ecg(train, validation=None, feature_extraction=None,
               combiner='cat', classifier=None, number_of_ecgs=2,
               dense_size=10):
    feature_extractor = load_model_from_experiment_result(**feature_extraction)

    if combiner == 'diff':
        combiner = difference_combiner
    else:
        combiner = Concatenate()

    inputs, x = stack_model(
        model=feature_extractor,
        combiner=combiner,
        stack_size=number_of_ecgs,
    )

    shape = train['x'].shape
    if 'features' in shape:
        feature_vector = Input(shape=shape['features'])
        inputs['features'] = feature_vector
        feature_vector = BatchNormalization()(feature_vector)
        x = Concatenate()([x, feature_vector])

    x = Dense(dense_size, activation="relu")(x)
    y = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)

    model = keras.Model(inputs, y)
    return model


def ffnn(train, validation=None, dense_layers=None, dropout=0):
    inp = {key: Input(shape=value) for key, value in train['x'].shape.items()}
    layers = []
    if 'ecg_0' in inp:
        layers.append(inp['ecg_0'])
    if 'ecg_1' in inp:
        layers.append(inp['ecg_1'])
    if 'flat_features' in inp:
        layers.append(BatchNormalization()(inp['flat_features']))

    if len(layers) > 1:
        x = Concatenate()(layers)
    else:
        x = layers[0]

    for size in dense_layers:
        x = Dense(size, activation='relu')(x)
        x = Dropout(dropout)(x)

    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)


def stack_model(model, combiner, stack_size=2):
    features = []
    inputs = {}

    for k in range(stack_size):
        feature = rename_model_layers(model, suffix=f'_ecg_{k}')
        features.append(feature.layers[-3].output)
        inputs[f'ecg_{k}'] = feature.input['ecg_0']

    if len(features) > 1:
        outputs = combiner(features)
    else:
        outputs = features[0]

    return inputs, outputs


def difference_combiner(features):
    diffs = [features[0]]
    for f in features[1:]:
        diffs.append(keras.layers.subtract([features[0], f]))

    return Concatenate()(diffs)


def rename_model_layers(model, prefix='', suffix='_bar'):
    config = rename_config(model.get_config(), prefix, suffix)
    new_model = model.from_config(config)
    new_model.set_weights(model.get_weights())
    return new_model


def rename_config(config, prefix, suffix):
    new_config = deepcopy(config)
    name_map = {
        layer['name']: prefix + layer['name'] + suffix
        for layer in config['layers']
    }

    for layer in new_config['layers']:
        old_name = layer['name']
        layer['config']['name'] = name_map[old_name]
        layer['name'] = name_map[old_name]
        for node in layer['inbound_nodes']:
            node[0][0] = name_map[node[0][0]]

    for layer in new_config['output_layers']:
        layer[0] = name_map[layer[0]]

    for layer in new_config['input_layers'].values():
        layer[0] = name_map[layer[0]]

    return new_config
