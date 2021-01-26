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


def ab_simple_lr(train=None, validation=None, **kwargs):
    inp = {key: Input(shape=value, name=key)
           for key, value in train['x'].shape.items()}
    to_cat = [value for key, value in sorted(inp.items())]
    layer = Concatenate()(to_cat)
    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(layer)
    return keras.Model(inp, output)
