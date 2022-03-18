# -*- coding: utf-8 -*-

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
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.regularizers import l2

from mim.util.logs import get_logger
from mim.models.load import (
    load_model_from_experiment_result,
    load_ribeiro_model
)

log = get_logger('simple_nn')


def logistic_regression_ab(train, validation=None):
    inp = {key: Input(shape=value) for key, value in train['x'].shape.items()}
    # ['log_dt', 'age', 'male', 'tnt_1']
    normalization = Normalization(axis=1)
    normalization.adapt(train['x']['flat_features'].as_numpy())
    cols = train['x']["flat_features"].columns
    if 'male' in cols:
        i = cols.index('male')
        w = normalization.get_weights()
        w[0][i] = 0
        w[1][i] = 1
        normalization.set_weights(w)
    layer = normalization(inp['flat_features'])
    output = Dense(1, activation="sigmoid", kernel_regularizer=None)(layer)
    return keras.Model(inp, output)


def logistic_regression(train, validation=None):
    inp = {key: Input(shape=value) for key, value in train['x'].shape.items()}
    layers = list(inp.values())
    if len(layers) > 1:
        x = Concatenate()(layers)
    else:
        x = layers[0]

    x = BatchNormalization()(x)
    output = Dense(1, activation="sigmoid", kernel_regularizer=None)(x)
    return keras.Model(inp, output)


def ecg_cnn(
        train,
        validation=None,
        cnn_kwargs=None,
        ecg_ffnn_kwargs=None,
        ecg_combiner='concatenate',
        ecg_comb_ffnn_kwargs=None,
        flat_ffnn_kwargs=None,
        final_ffnn_kwargs=None):
    inp = {key: Input(shape=value) for key, value in train['x'].shape.items()}
    ecg_layers = []
    if 'ecg_0' in inp:
        ecg_layers.append(_ecg_network(inp['ecg_0'], **cnn_kwargs))
    if 'ecg_1' in inp:
        ecg_layers.append(_ecg_network(inp['ecg_1'], **cnn_kwargs))

    return _ecg_and_flat_feature_combiner(
        inp=inp,
        ecg_layers=ecg_layers,
        ecg_ffnn_kwargs=ecg_ffnn_kwargs,
        ecg_comb_ffnn_kwargs=ecg_comb_ffnn_kwargs,
        ecg_combiner=ecg_combiner,
        flat_ffnn_kwargs=flat_ffnn_kwargs,
        final_ffnn_kwargs=final_ffnn_kwargs,
        output_size=len(train['y'].columns)
    )


def _ecg_and_flat_feature_combiner(
        inp, ecg_layers, ecg_ffnn_kwargs, ecg_comb_ffnn_kwargs, ecg_combiner,
        flat_ffnn_kwargs, final_ffnn_kwargs, output_size
):
    assert len(ecg_layers) >= 1
    if ecg_ffnn_kwargs is not None:
        ecg_layers = [ffnn_helper(x, **ecg_ffnn_kwargs) for x in ecg_layers]

    if len(ecg_layers) > 1:
        if ecg_combiner == 'difference':
            x = difference_combiner(ecg_layers)
        else:
            x = Concatenate()(ecg_layers)
    else:
        x = ecg_layers[0]

    if ecg_comb_ffnn_kwargs is not None:
        x = ffnn_helper(x, **ecg_comb_ffnn_kwargs)

    if 'flat_features' in inp:
        flat_features = BatchNormalization()(inp['flat_features'])
        if flat_ffnn_kwargs is not None:
            flat_features = ffnn_helper(flat_features, **flat_ffnn_kwargs)
        x = Concatenate()([x, flat_features])

    if final_ffnn_kwargs is not None:
        x = ffnn_helper(x, **final_ffnn_kwargs)

    output = Dense(output_size, activation="sigmoid",
                   kernel_regularizer="l2")(x)
    return keras.Model(inp, output)


def ffnn_helper(x, sizes, dropouts, batch_norms, activation='relu',
                activity_regularizer=None,
                activity_regularizers=None,
                kernel_regularizer=None,
                kernel_regularizers=None,
                bias_regularizer=None,
                bias_regularizers=None):
    num_layers = len(sizes)
    if activity_regularizers is None:
        if activity_regularizer is None:
            activity_regularizer = 0.0
        activity_regularizers = num_layers*[activity_regularizer]
    if kernel_regularizers is None:
        if kernel_regularizer is None:
            kernel_regularizer = 0.0
        kernel_regularizers = num_layers*[kernel_regularizer]
    if bias_regularizers is None:
        if bias_regularizer is None:
            bias_regularizer = 0.0
        bias_regularizers = num_layers*[bias_regularizer]

    assert _all_lists_have_same_length(
        [sizes, dropouts, batch_norms, activity_regularizers,
         kernel_regularizers, bias_regularizers]
    )

    for layer in range(num_layers):
        x = Dense(
            sizes[layer],
            activation=activation,
            activity_regularizer=l2(activity_regularizers[layer]),
            kernel_regularizer=l2(kernel_regularizers[layer]),
            bias_regularizer=l2(bias_regularizers[layer])
        )(x)

        if batch_norms[layer]:
            x = BatchNormalization()(x)
        if dropouts[layer] > 0:
            x = Dropout(dropouts[layer])(x)

    return x


def _ecg_network(
        x,
        down_sample=False,
        num_layers=2,
        dropout=0.3,
        dropouts=None,
        filter_first=16,
        filter_last=16,
        filters=None,
        kernel_first=5,
        kernel_last=5,
        kernels=None,
        batch_norm=True,
        batch_norms=None,
        weight_decay=None,
        weight_decays=None,
        pool_size=None,
        pool_sizes=None):

    if down_sample:
        x = AveragePooling1D(2, padding='same')(x)

    if pool_sizes is None:
        if pool_size is None:
            pool_size = _calculate_appropriate_pool_size(
                input_size=x.shape[1],
                num_pools=num_layers,
                minimum_output_size=4
            )
        pool_sizes = num_layers * [pool_size]

    if filters is None:
        if filter_first is not None and filter_last is not None:
            filters = list(map(
                round, np.linspace(filter_first, filter_last, num_layers)))
        else:
            raise ValueError('Must specify either filters or both '
                             'filter_first and filter_last. ')

    if kernels is None:
        if kernel_first is not None and kernel_last is not None:
            kernels = list(map(
                round, np.linspace(kernel_first, kernel_last, num_layers)))
        else:
            raise ValueError('Must specify either kernels or both '
                             'kernel_first and kernel_last. ')

    if dropouts is None:
        dropouts = num_layers * [dropout]

    if weight_decays is None:
        if weight_decay is None:
            weight_decay = 0.01
        weight_decays = num_layers * [weight_decay]

    if batch_norms is None:
        batch_norms = num_layers * [batch_norm]

    assert _all_lists_have_same_length(
        [filters, kernels, weight_decays, batch_norms, pool_sizes, dropouts],
        expected_length=num_layers
    )

    for layer in range(num_layers):
        x = Conv1D(
            filters=filters[layer],
            kernel_size=kernels[layer],
            kernel_regularizer=l2(weight_decays[layer]),
            padding='same')(x)
        if batch_norms[layer]:
            x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool1D(pool_size=pool_sizes[layer])(x)
        x = Dropout(dropouts[layer])(x)

    x = Flatten()(x)
    return x


def _all_lists_have_same_length(lists, expected_length=None):
    if expected_length is None:
        expected_length = len(lists[0])

    return all(map(lambda x: len(x) == expected_length, lists))


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

    x = Dense(dense_size, activation="relu")(x)

    shape = train['x'].shape
    if 'flat_features' in shape:
        feature_vector = Input(shape=shape['flat_features'])
        inputs['flat_features'] = feature_vector
        feature_vector = BatchNormalization()(feature_vector)
        x = Concatenate()([x, feature_vector])

    y = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)

    model = keras.Model(inputs, y)
    return model


def pretrained_resnet(
        train, validation=None, freeze_resnet=False, ecg_ffnn_kwargs=None,
        ecg_combiner='concatenate', ecg_comb_ffnn_kwargs=None,
        flat_ffnn_kwargs=None, final_ffnn_kwargs=None
):
    """
    Load a pre-trained ResNet model for each input ECG, and plug this into
    the serial-ecg architecture with additional, optional ffnns.
    """
    data_shape = train['x'].shape
    inp = {}
    ecg_layers = []
    for ecg in [key for key in data_shape if key.startswith('ecg')]:
        resnet_input, resnet_output = load_ribeiro_model(
            freeze_resnet, suffix=f"_{ecg}")
        inp[ecg] = resnet_input
        ecg_layers.append(resnet_output)

    if 'flat_features' in data_shape:
        inp['flat_features'] = Input(shape=data_shape['flat_features'])

    return _ecg_and_flat_feature_combiner(
        inp=inp,
        ecg_layers=ecg_layers,
        ecg_ffnn_kwargs=ecg_ffnn_kwargs,
        ecg_comb_ffnn_kwargs=ecg_comb_ffnn_kwargs,
        ecg_combiner=ecg_combiner,
        flat_ffnn_kwargs=flat_ffnn_kwargs,
        final_ffnn_kwargs=final_ffnn_kwargs,
        output_size=len(train['y'].columns)
    )


def ffnn(
        train,
        validation=None,
        ecg_ffnn_kwargs=None,
        ecg_combiner='concatenate',
        ecg_comb_ffnn_kwargs=None,
        flat_ffnn_kwargs=None,
        final_ffnn_kwargs=None,
):
    inp = _make_input(train['x'].shape)
    ecg_layers = []
    if 'ecg_0' in inp:
        ecg_layers.append(inp['ecg_0'])
    if 'ecg_1' in inp:
        ecg_layers.append(inp['ecg_1'])
    if 'forberg_ecg_0' in inp:
        ecg_layers.append(inp['forberg_ecg_0'])
    if 'forberg_ecg_1' in inp:
        ecg_layers.append(inp['forberg_ecg_1'])
    if 'forberg_diff' in inp:
        ecg_layers.append(inp['forberg_diff'])

    return _ecg_and_flat_feature_combiner(
        inp=inp,
        ecg_layers=ecg_layers,
        ecg_ffnn_kwargs=ecg_ffnn_kwargs,
        ecg_comb_ffnn_kwargs=ecg_comb_ffnn_kwargs,
        ecg_combiner=ecg_combiner,
        flat_ffnn_kwargs=flat_ffnn_kwargs,
        final_ffnn_kwargs=final_ffnn_kwargs,
        output_size=len(train['y'].columns)
    )


def _make_input(shape):
    if isinstance(shape, dict):
        return {key: _make_input(value) for key, value in shape.items()}
    else:
        return Input(shape=shape)


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
