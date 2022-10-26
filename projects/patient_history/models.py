from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Concatenate,
    Dense, LSTM, Dropout
)

from mim.models.util import mlp_helper


def _make_input(shape):
    if isinstance(shape, dict):
        return {key: _make_input(value) for key, value in shape.items()}
    else:
        return Input(shape=shape)


def mlp1(train, validation=None, history_mlp_kwargs=None,
         lisa_mlp_kwargs=None, basic_mlp_kwargs=None,
         final_mlp_kwargs=None):
    inp = _make_input(train.feature_tensor_shape)
    layers = []
    if 'history' in inp:
        history = inp['history']
        layers.append(mlp_helper(history, **history_mlp_kwargs))
    if 'lisa' in inp:
        layers.append(mlp_helper(inp['lisa'], **lisa_mlp_kwargs))
    if 'basic' in inp:
        if basic_mlp_kwargs:
            layers.append(mlp_helper(inp['basic'], **basic_mlp_kwargs))
        else:
            layers.append(inp['basic'])

    if len(layers) > 1:
        x = Concatenate()(layers)
    else:
        x = layers[0]

    if final_mlp_kwargs:
        x = mlp_helper(x, **final_mlp_kwargs)

    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)


def mlp2(train, validation=None, mlp_kwargs=None):
    inp = _make_input(train.feature_tensor_shape)

    x = Concatenate()(inp.values())
    if mlp_kwargs:
        x = mlp_helper(x, **mlp_kwargs)

    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)


def simple_lstm(train, validation=None, **kwargs):
    inp = Input(shape=train.feature_tensor_shape, ragged=True)

    x = LSTM(11)(inp)
    x = Dense(10, activation='relu')(x)
    x = Dropout(0.2)(x)

    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)
