import numpy as np
from tensorflow import keras
from keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    Activation,
    Flatten,
    Dense
)

from mim.models.util import (
    cnn_helper, mlp_helper, ResidualUnit, ResidualUnitV2
)


def cnn(
        train,
        validation=None,
        cnn_kwargs=None,
        ffnn_kwargs=None,
):
    # inp = {
    #     key: Input(shape=value)
    #     for key, value in train.feature_tensor_shape.items()
    # }

    inp = Input(shape=train.feature_tensor_shape)
    x = cnn_helper(inp, **cnn_kwargs)
    if ffnn_kwargs:
        x = mlp_helper(x, **ffnn_kwargs)
    output = Dense(units=1, activation='sigmoid', kernel_regularizer='l2')(x)

    return keras.Model(inp, output)


def resnet_v1(
        train, validation=None, residual_kwargs=None
):
    signal = Input(shape=(4096, 8), dtype=np.float16, name='signal')
    x = Conv1D(64, 17, padding='same', use_bias=False,
               kernel_initializer='he_normal')(signal)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x, y = ResidualUnit(1024, 128, **residual_kwargs)([x, x])
    x, y = ResidualUnit(256, 196, **residual_kwargs)([x, y])
    x, y = ResidualUnit(64, 256, **residual_kwargs)([x, y])
    x, _ = ResidualUnit(16, 320, **residual_kwargs)([x, y])
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)
    model = keras.Model(signal, output)
    return model


def resnet_v2(
        train, validation=None, filters=64, residual_kwargs=None
):
    if residual_kwargs is None:
        residual_kwargs = {}

    signal = Input(shape=(4096, 8), dtype=np.float16, name='signal')
    x = Conv1D(64, 17, padding='same', use_bias=False,
               kernel_initializer='he_normal')(signal)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x, y = ResidualUnitV2(2048, filters, **residual_kwargs)([x, x])
    x, y = ResidualUnitV2(2048, filters, **residual_kwargs)([x, y])
    x, y = ResidualUnitV2(1024, filters, **residual_kwargs)([x, y])

    x, y = ResidualUnitV2(512, 2*filters, **residual_kwargs)([x, y])
    x, y = ResidualUnitV2(512, 2*filters, **residual_kwargs)([x, y])
    x, y = ResidualUnitV2(256, 2*filters, **residual_kwargs)([x, y])

    x, y = ResidualUnitV2(128, 4*filters, **residual_kwargs)([x, y])
    x, y = ResidualUnitV2(128, 4*filters, **residual_kwargs)([x, y])
    x, y = ResidualUnitV2(64, 4*filters, **residual_kwargs)([x, y])

    x, y = ResidualUnitV2(32, 8*filters, **residual_kwargs)([x, y])
    x, y = ResidualUnitV2(32, 8*filters, **residual_kwargs)([x, y])
    x, _ = ResidualUnitV2(16, 8*filters, **residual_kwargs)([x, y])

    x = Flatten()(x)
    output = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)
    model = keras.Model(signal, output)
    return model


def mlp(
        train,
        validation=None,
        mlp_kwargs=None,
):
    inp = Input(shape=train.feature_tensor_shape)
    x = mlp_helper(inp, **mlp_kwargs)
    output = Dense(units=1, activation='sigmoid', kernel_regularizer='l2')(x)
    return keras.Model(inp, output)
