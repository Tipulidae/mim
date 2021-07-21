# -*- coding: utf-8 -*-
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input,
    Activation,
    Add,
    Dense,
    Flatten,
    Conv1D,
    MaxPool1D,
    Dropout,
    BatchNormalization,
    Concatenate,
    Lambda
)
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.regularizers import l2


def ab_simple_lr(train=None, validation=None, **kwargs):
    inp = {key: Input(shape=value, name=key)
           for key, value in train['x'].shape.items()}
    # to_cat = [value for key, value in sorted(inp.items())]
    to_cat = [inp['categorical']]
    normalization = Normalization(axis=1)
    normalization.adapt(train['x']['numeric'].as_numpy())
    to_cat.append(normalization(inp['numeric']))
    layer = Concatenate()(to_cat)
    output = Dense(1, activation="sigmoid")(layer)
    return keras.Model(inp, output)


def ab_simple_one_hidden_layer(train=None, **kwargs):
    inp = {key: Input(shape=value, name=key)
           for key, value in train['x'].shape.items()}
    normalization = Normalization(axis=1)
    normalization.adapt(train['x']['numeric'].as_numpy())
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
            skip_basic=False,
            ecg_normalization_layer=None,
            **kwargs):
    inp = {key: Input(shape=value, name=key)
           for key, value in train['x'].shape.items()}

    # Basic
    if not skip_basic:
        to_cat = [inp['categorical']]
        normalization = Normalization(axis=1, name="NormalizeNumeric")
        normalization.adapt(train['x']['numeric'].as_numpy())
        to_cat.append(normalization(inp['numeric']))
        basic_layer = Concatenate(name="CatBasic")(to_cat)
    else:
        basic_layer = None

    # ECG Processing
    layer = inp["ecg"]
    if ecg_normalization_layer == "BatchNormalization":
        layer = BatchNormalization()(layer)
    elif ecg_normalization_layer == "PredefinedLambda":
        m = 4.057771e-05
        s = 0.0001882498
        normalization_layer = Lambda(lambda v: (v - m) / s)
        layer = normalization_layer(layer)

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
    if basic_layer:
        layer = Concatenate(name="Merge")([layer, basic_layer])
    layer = Dense(units=final_dense_neurons,
                  activation=activation,
                  name="FinalDense")(layer)
    if final_dense_dropout > 0:
        layer = Dropout(final_dense_dropout,
                        name="FinalDropout")(layer)
    output = Dense(1, activation="sigmoid")(layer)
    return keras.Model(inp, output)


def get_stanford_cnn(train,
                     conv_filters=16,
                     conv_kernel_size=16,
                     conv_weight_decay=0.001,
                     ecg_dense_neurons=25,
                     ecg_normalization_layer=None
                     ):
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import Lambda

    inp = {key: Input(shape=value, name=key)
           for key, value in train['x'].shape.items()}

    input_ecg = inp["ecg"]
    if ecg_normalization_layer:
        input_ecg = ecg_normalization_layer(input_ecg)

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    filters = conv_filters
    layer = Conv1D(filters=filters, kernel_size=conv_kernel_size,
                   kernel_initializer="he_normal", strides=1,
                   padding="same", kernel_regularizer=l2(conv_weight_decay))(
        input_ecg)
    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    ##
    shortcut = MaxPool1D(pool_size=1, padding="same")(layer)
    layer = Conv1D(filters=filters, kernel_size=conv_kernel_size,
                   kernel_initializer="he_normal",
                   strides=1, padding="same",
                   kernel_regularizer=l2(conv_weight_decay))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    layer = Dropout(0.2)(layer)
    layer = Conv1D(filters=filters, kernel_size=conv_kernel_size,
                   kernel_initializer="he_normal",
                   strides=1, padding="same",
                   kernel_regularizer=l2(conv_weight_decay))(layer)
    layer = Add()([layer, shortcut])
    for i in range(14):
        subsample = 1 if (i % 2) == 0 else 2
        shortcut = MaxPool1D(pool_size=subsample, padding="same")(layer)
        if (i + 1) % 4 == 0:
            filters = filters * 2
            shortcut = Lambda(
                zeropad, output_shape=zeropad_output_shape)(shortcut)
        layer = BatchNormalization()(layer)
        layer = Activation("relu")(layer)
        layer = Dropout(0.2)(layer)
        layer = Conv1D(filters=filters, kernel_size=conv_kernel_size,
                       kernel_initializer="he_normal",
                       strides=subsample, padding="same",
                       kernel_regularizer=l2(conv_weight_decay))(layer)
        layer = BatchNormalization()(layer)
        layer = Activation("relu")(layer)
        layer = Dropout(0.2)(layer)
        layer = Conv1D(filters=filters, kernel_size=conv_kernel_size,
                       kernel_initializer="he_normal",
                       strides=1, padding="same",
                       kernel_regularizer=l2(conv_weight_decay))(layer)
        layer = Add()([layer, shortcut])

    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    layer = Flatten()(layer)
    layer = Dense(ecg_dense_neurons, activation="relu")(layer)
    output = Dense(1, activation="sigmoid")(layer)
    return keras.Model(inp, output)
