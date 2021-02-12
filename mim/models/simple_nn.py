import os
import math

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv1D,
    MaxPool1D,
    AveragePooling1D,
    Dropout,
    BatchNormalization,
    Concatenate,
    ReLU,
    Lambda,
)


def super_basic_cnn(train, validation=None, dropout=0, filters=32,
                    kernel_size=16, pool_size=8, dense=False,
                    dense_activation=None):
    inp = {key: Input(shape=value) for key, value in train['x'].shape.items()}
    m = 4.057771e-05
    s = 0.0001882498
    x = inp['ecg']
    x = Lambda(lambda v: (v - m) / s)(x)
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        kernel_regularizer="l2",
        padding='same')(x)
    x = ReLU()(x)
    x = MaxPool1D(pool_size=pool_size)(x)
    x = Dropout(dropout)(x)

    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        kernel_regularizer="l2",
        padding='same')(x)
    x = ReLU()(x)
    x = MaxPool1D(pool_size=pool_size)(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)

    if dense:
        x = Dense(10, activation=dense_activation)(x)
        x = Dropout(dropout)(x)

    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)


def sequential_cnn(train, validation=None, dropout=0, filter_first=16,
                   filter_last=16, kernel_first=5, kernel_last=5, num_layers=2,
                   dense=True, batch_norm=True):
    inp = {key: Input(shape=value) for key, value in train['x'].shape.items()}
    # m = 4.057771e-05
    # s = 0.0001882498
    m = 7.1811132e-06
    s = 0.0002694354
    x = inp['ecg']
    x = Lambda(lambda v: (v - m) / s)(x)
    x = AveragePooling1D(2, padding='same')(x)

    pool_size = math.floor(500 ** (1 / num_layers))
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
        x = Dense(10, activation='relu')(x)
        x = Dropout(dropout)(x)

    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)


def basic_cnn2(train, validation=None, dropout=0, layers=None,
               hidden_layer=None):
    inp = {key: Input(shape=value) for key, value in train['x'].shape.items()}
    x = inp['ecg']
    for layer in layers:
        x = Conv1D(
            filters=layer['filters'],
            kernel_size=layer['kernel_size'],
            kernel_regularizer="l2",
            padding='same')(x)
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


def basic_cnn(train, validation=None, num_conv_layers=2, dropout=0.3,
              filters=32, kernel_size=16):
    inp = {key: Input(shape=value) for key, value in train['x'].shape.items()}
    layers = []
    if 'ecg' in inp:
        layers.append(
            _ecg_network(
                inp['ecg'],
                num_conv_layers,
                dropout=dropout,
                filters=filters,
                kernel_size=kernel_size
            )
        )
    if 'old_ecg' in inp:
        layers.append(
            _ecg_network(
                inp['old_ecg'],
                num_conv_layers,
                dropout=dropout,
                filters=filters,
                kernel_size=kernel_size
            )
        )
    if 'features' in inp:
        layers.append(BatchNormalization()(inp['features']))

    if len(layers) > 1:
        x = Concatenate()(layers)
    else:
        x = layers[0]

    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)


def _ecg_network(ecg, num_conv_layers, dropout=0.2, filters=32,
                 kernel_size=16):
    ecg = BatchNormalization()(ecg)
    for _ in range(num_conv_layers):
        ecg = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            kernel_regularizer="l2",
            padding='same')(ecg)
        ecg = BatchNormalization()(ecg)
        ecg = ReLU()(ecg)
        ecg = MaxPool1D(pool_size=16)(ecg)
        ecg = Dropout(dropout)(ecg)

    return Flatten()(ecg)


def basic_ff():
    inp = Input(shape=(128, ))
    x = Flatten()(inp)
    x = Dense(32, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)
    model = keras.Model(inp, output)
    return model


def load_keras_model(base_path, split_number, **kwargs):
    path = os.path.join(
        base_path,
        f"split_{split_number}",
        "last.ckpt"
    )
    return keras.models.load_model(filepath=path)
