# -*- coding: utf-8 -*-
from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Dense,
    # Flatten,
    # Conv1D,
    # MaxPool1D,
    # Dropout,
    # BatchNormalization,
    Concatenate,
    # ReLU
)
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
