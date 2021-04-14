# -*- coding: utf-8 -*-
from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv1D,
    MaxPool1D,
    Dropout,
    # BatchNormalization,
    Concatenate
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers.experimental.preprocessing import Normalization


def ab_simple_lr(train=None, validation=None, **kwargs):
    inp = {key: Input(shape=value, name=key)
           for key, value in train['x'].shape.items()}
    # to_cat = [value for key, value in sorted(inp.items())]
    to_cat = [inp['categorical']]
    normalization = Normalization(axis=1)
    normalization.adapt(train['x']['numeric'].as_numpy)
    to_cat.append(normalization(inp['numeric']))
    layer = Concatenate()(to_cat)
    output = Dense(1, activation="sigmoid")(layer)
    return keras.Model(inp, output)


def ab_simple_one_hidden_layer(train=None, **kwargs):
    inp = {key: Input(shape=value, name=key)
           for key, value in train['x'].shape.items()}
    normalization = Normalization(axis=1)
    normalization.adapt(train['x']['numeric'].as_numpy)
    normalized = normalization(inp['numeric'])
    concatenated = Concatenate()([normalized, inp['categorical']])
    layer = Dense(kwargs["hidden_layer_n"], activation="relu",
                  kernel_regularizer=l2(l2=kwargs["l2"]),
                  bias_regularizer=l2(l2=kwargs["l2"])
                  )(concatenated)
    if "dense_dropout" in kwargs and 0 < kwargs["dense_dropout"] < 1:
        layer = Dropout(kwargs["dense_dropout"])(layer)
    output = Dense(1, activation="sigmoid")(layer)
    return keras.Model(inp, output)


def dyn_cnn(train=None, validation=None,
            conv_dropout=None,
            conv_filters=None,
            conv_kernel_size=None,
            conv_pool_size=None,
            conv_weight_decay=None,
            conv_final_dense_neurons=None,
            final_dense_neurons=None,
            final_dense_dropout=None,
            activation=None,
            **kwargs):
    inp = {key: Input(shape=value, name=key)
           for key, value in train['x'].shape.items()}

    # Basic
    to_cat = [inp['categorical']]
    normalization = Normalization(axis=1, name="NormalizeNumeric")
    normalization.adapt(train['x']['numeric'].as_numpy)
    to_cat.append(normalization(inp['numeric']))
    basic_layer = Concatenate(name="CatBasic")(to_cat)

    # ECG Processing
    layer = inp["ecg"]
    assert len(conv_dropout) == len(conv_filters)
    assert len(conv_dropout) == len(conv_kernel_size)
    assert len(conv_dropout) == len(conv_pool_size)
    assert len(conv_dropout) == len(conv_weight_decay)
    for i in range(len(conv_dropout)):
        layer = Conv1D(filters=conv_filters[i],
                       kernel_size=conv_kernel_size[i],
                       activation=activation,
                       kernel_regularizer=l2(conv_weight_decay[i]),
                       name=f"Conv{i}")(layer)
        layer = MaxPool1D(pool_size=conv_pool_size[i],
                          name=f"MaxPool{i}")(layer)
        if conv_dropout[i] > 0:
            layer = Dropout(conv_dropout[i],
                            name=f"ConvDropout{i}")(layer)
    layer = Flatten(name="FlattenECG")(layer)
    layer = Dense(units=conv_final_dense_neurons,
                  activation=activation,
                  name="ECGDense")(layer)

    # Merging
    layer = Concatenate(name="Merge")([layer, basic_layer])
    layer = Dense(units=final_dense_neurons,
                  activation=activation,
                  name="FinalDense")(layer)
    if final_dense_dropout > 0:
        layer = Dropout(final_dense_dropout,
                        name="FinalDropout")(layer)
    output = Dense(1, activation="sigmoid")(layer)
    return keras.Model(inp, output)
