# -*- coding: utf-8 -*-

from copy import deepcopy

from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    BatchNormalization,
    Concatenate
)
from tensorflow.keras.layers.experimental.preprocessing import Normalization

from mim.models.util import cnn_helper, ffnn_helper
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


def ptbxl_cnn(
        train,
        validation=None,
        cnn_kwargs=None,
        ffnn_kwargs=None,
        final_ffnn_kwargs=None
):
    if final_ffnn_kwargs is None:
        final_ffnn_kwargs = {}

    inp = Input(shape=train['x'].shape)
    x = cnn_helper(inp, **cnn_kwargs)
    x = ffnn_helper(x, **ffnn_kwargs)

    output_layers = []

    for name in train['y'].columns:
        y = x
        if name in final_ffnn_kwargs:
            y = ffnn_helper(x, **final_ffnn_kwargs[name])

        output_layers.append(
            Dense(
                units=1,
                activation='sigmoid' if name == 'sex' else None,
                kernel_regularizer='l2',
                name=name
            )(y)
        )

    if len(output_layers) > 1:
        output = output_layers
    else:
        output = output_layers[0]

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
        ecg_layers.append(cnn_helper(inp['ecg_0'], **cnn_kwargs))
    if 'ecg_1' in inp:
        ecg_layers.append(cnn_helper(inp['ecg_1'], **cnn_kwargs))

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
