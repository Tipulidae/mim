from tensorflow import keras
from tensorflow.keras.layers import Input, Concatenate, BatchNormalization, \
    Dense

# This is just to make PyCharm resolve the import. Otherwise, I should be
# able to import it straight from tf.keras.layers.
from tensorflow.python.keras.layers import Normalization

from mim.models.load import load_ribeiro_model
from mim.models.util import cnn_helper, mlp_helper


def ecg_cnn(
        train,
        validation=None,
        cnn_kwargs=None,
        ecg_ffnn_kwargs=None,
        ecg_combiner='concatenate',
        ecg_comb_ffnn_kwargs=None,
        flat_ffnn_kwargs=None,
        final_ffnn_kwargs=None):
    inp = {
        key: Input(shape=value)
        for key, value in train.feature_tensor_shape.items()
    }
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
        output_size=train.output_size
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
    inp = _make_input(train.feature_tensor_shape)
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
        output_size=train.output_size
    )


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


def logistic_regression_ab(train, validation=None):
    inp = {
        key: Input(shape=value)
        for key, value in train.feature_tensor_shape.items()}
    # ['log_dt', 'age', 'male', 'tnt_1']
    normalization = Normalization(axis=1)
    normalization.adapt(train.x(can_use_tf_dataset=False))
    cols = train.feature_names['flat_features']
    if 'male' in cols:
        i = cols.index('male')
        w = normalization.get_weights()
        w[0][i] = 0
        w[1][i] = 1
        normalization.set_weights(w)
    layer = normalization(inp['flat_features'])
    output = Dense(1, activation="sigmoid", kernel_regularizer=None)(layer)
    return keras.Model(inp, output)


def pretrained_resnet(
        train, validation=None, freeze_resnet=False, ecg_ffnn_kwargs=None,
        ecg_combiner='concatenate', ecg_comb_ffnn_kwargs=None,
        flat_ffnn_kwargs=None, final_ffnn_kwargs=None
):
    """
    Load a pre-trained ResNet model for each input ECG, and plug this into
    the serial-ecg architecture with additional, optional ffnns.
    """
    feature_shape = train.feature_tensor_shape
    inp = {}
    ecg_layers = []
    for ecg in [key for key in feature_shape if key.startswith('ecg')]:
        resnet_input, resnet_output = load_ribeiro_model(
            freeze_resnet, suffix=f"_{ecg}")
        inp[ecg] = resnet_input
        ecg_layers.append(resnet_output)

    if 'flat_features' in feature_shape:
        inp['flat_features'] = Input(shape=feature_shape['flat_features'])

    return _ecg_and_flat_feature_combiner(
        inp=inp,
        ecg_layers=ecg_layers,
        ecg_ffnn_kwargs=ecg_ffnn_kwargs,
        ecg_comb_ffnn_kwargs=ecg_comb_ffnn_kwargs,
        ecg_combiner=ecg_combiner,
        flat_ffnn_kwargs=flat_ffnn_kwargs,
        final_ffnn_kwargs=final_ffnn_kwargs,
        output_size=train.output_size
    )


def _make_input(shape):
    if isinstance(shape, dict):
        return {key: _make_input(value) for key, value in shape.items()}
    else:
        return Input(shape=shape)


def _ecg_and_flat_feature_combiner(
        inp, ecg_layers, ecg_ffnn_kwargs, ecg_comb_ffnn_kwargs, ecg_combiner,
        flat_ffnn_kwargs, final_ffnn_kwargs, output_size
):
    assert len(ecg_layers) >= 1
    if ecg_ffnn_kwargs is not None:
        ecg_layers = [mlp_helper(x, **ecg_ffnn_kwargs) for x in ecg_layers]

    if len(ecg_layers) > 1:
        if ecg_combiner == 'difference':
            x = difference_combiner(ecg_layers)
        else:
            x = Concatenate()(ecg_layers)
    else:
        x = ecg_layers[0]

    if ecg_comb_ffnn_kwargs is not None:
        x = mlp_helper(x, **ecg_comb_ffnn_kwargs)

    if 'flat_features' in inp:
        flat_features = BatchNormalization()(inp['flat_features'])
        if flat_ffnn_kwargs is not None:
            flat_features = mlp_helper(flat_features, **flat_ffnn_kwargs)
        x = Concatenate()([x, flat_features])

    if final_ffnn_kwargs is not None:
        x = mlp_helper(x, **final_ffnn_kwargs)

    output = Dense(output_size, activation="sigmoid",
                   kernel_regularizer="l2")(x)
    return keras.Model(inp, output)


def difference_combiner(features):
    diffs = [features[0]]
    for f in features[1:]:
        diffs.append(keras.layers.subtract([features[0], f]))

    return Concatenate()(diffs)
