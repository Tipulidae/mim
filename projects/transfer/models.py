import numpy as np
from tensorflow import keras
from keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    Activation,
    Flatten,
    Dense,
    Concatenate
)

from mim.models.load import (
    load_model_from_experiment_result, load_ribeiro_model
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
    inp = Input(shape=train.feature_tensor_shape)
    x = cnn_helper(inp, **cnn_kwargs)
    if ffnn_kwargs:
        x = mlp_helper(x, **ffnn_kwargs)
    output = Dense(units=1, activation='sigmoid', kernel_regularizer='l2')(x)

    return keras.Model(inp, output)


def pretrained(train, validation=None, from_xp=None, ecg_mlp_kwargs=None,
               flat_mlp_kwargs=None, final_mlp_kwargs=None):
    inp = {}
    if 'flat_features' in train.feature_tensor_shape:
        inp['flat_features'] = Input(
            shape=train.feature_tensor_shape['flat_features'],
        )
    ecg_inp, x = load_model_from_experiment_result(**from_xp)
    inp['ecg'] = ecg_inp

    if ecg_mlp_kwargs is not None:
        x = mlp_helper(x, **ecg_mlp_kwargs)

    if flat_mlp_kwargs is not None:
        flat_features = mlp_helper(inp['flat_features'], **flat_mlp_kwargs)
        x = Concatenate()([x, flat_features])

    if final_mlp_kwargs is not None:
        x = mlp_helper(x, **final_mlp_kwargs)
    output = Dense(units=1, activation='sigmoid', kernel_regularizer='l2')(x)
    return keras.Model(inp, output)


def pretrained_parallel(
        train, validation=None, from_xp1=None, from_xp2=None,
        ecg_mlp_kwargs=None, flat_mlp_kwargs=None, final_mlp_kwargs=None):
    inp = {}
    if 'flat_features' in train.feature_tensor_shape:
        inp['flat_features'] = Input(
            shape=train.feature_tensor_shape['flat_features'],
        )
    ecg_inp1, x1 = load_model_from_experiment_result(**from_xp1)
    ecg_inp2, x2 = load_model_from_experiment_result(**from_xp2)
    inp['ecg'] = Input(shape=(4096, 8), dtype=np.float16, name='signal')

    m1 = keras.Model(ecg_inp1, x1, name='model1')
    m2 = keras.Model(ecg_inp2, x2, name='model2')

    x1 = m1(inp['ecg'])
    x2 = m2(inp['ecg'])
    x = Concatenate()([x1, x2])

    if ecg_mlp_kwargs is not None:
        x = mlp_helper(x, **ecg_mlp_kwargs)

    if flat_mlp_kwargs is not None:
        flat_features = mlp_helper(inp['flat_features'], **flat_mlp_kwargs)
        x = Concatenate()([x, flat_features])

    if final_mlp_kwargs is not None:
        x = mlp_helper(x, **final_mlp_kwargs)
    output = Dense(units=1, activation='sigmoid', kernel_regularizer='l2')(x)
    return keras.Model(inp, output)


def ribeiros_resnet(train, validation=None, final_mlp_kwargs=None):
    inp, x = load_ribeiro_model(freeze_resnet=True, suffix='_rn')
    if final_mlp_kwargs is not None:
        x = mlp_helper(x, **final_mlp_kwargs)

    output = Dense(units=1, activation='sigmoid', kernel_regularizer='l2')(x)
    return keras.Model({'ecg': inp}, output)


def resnet_v1(train, validation=None, residual_kwargs=None,
              flat_mlp_kwargs=None, ecg_mlp_kwargs=None,
              final_mlp_kwargs=None):
    if residual_kwargs is None:
        residual_kwargs = {}

    inp = {}
    if 'flat_features' in train.feature_tensor_shape:
        inp['flat_features'] = Input(
            shape=train.feature_tensor_shape['flat_features'],
        )
    if 'ecg' in train.feature_tensor_shape:
        inp['ecg'] = Input(shape=(4096, 8), dtype=np.float16, name='signal')

    # signal = Input(shape=(4096, 8), dtype=np.float16, name='signal')
    x = Conv1D(64, 17, padding='same', use_bias=False,
               kernel_initializer='he_normal')(inp['ecg'])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x, y = ResidualUnit(1024, 128, **residual_kwargs)([x, x])
    x, y = ResidualUnit(256, 196, **residual_kwargs)([x, y])
    x, y = ResidualUnit(64, 256, **residual_kwargs)([x, y])
    x, _ = ResidualUnit(16, 320, **residual_kwargs)([x, y])
    x = Flatten()(x)

    if ecg_mlp_kwargs is not None:
        x = mlp_helper(x, **ecg_mlp_kwargs)
    if 'flat_features' in inp:
        flat_features = inp['flat_features']
        if flat_mlp_kwargs is not None:
            flat_features = mlp_helper(flat_features, **flat_mlp_kwargs)
        x = Concatenate()([x, flat_features])
    if final_mlp_kwargs is not None:
        x = mlp_helper(x, **final_mlp_kwargs)

    output_layers = []
    for name in train.target_columns:
        y = x
        output_layers.append(
            Dense(
                units=1,
                activation='sigmoid' if name == 'sex' else None,
                kernel_initializer='he_normal',
                name=name
            )(y)
        )

    if len(output_layers) > 1:
        output = output_layers
    else:
        output = output_layers[0]

    model = keras.Model(inp, output)
    return model


def resnet_v2(
        train, validation=None, filters=64, residual_kwargs=None,
        flat_mlp_kwargs=None, ecg_mlp_kwargs=None, final_mlp_kwargs=None
):
    if residual_kwargs is None:
        residual_kwargs = {}

    inp = {}
    if 'flat_features' in train.feature_tensor_shape:
        inp['flat_features'] = Input(
            shape=train.feature_tensor_shape['flat_features'],
        )
    if 'ecg' in train.feature_tensor_shape:
        inp['ecg'] = Input(shape=(4096, 8), dtype=np.float16, name='signal')

    x = Conv1D(64, 17, padding='same', use_bias=False,
               kernel_initializer='he_normal')(inp['ecg'])
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

    if ecg_mlp_kwargs is not None:
        x = mlp_helper(x, **ecg_mlp_kwargs)
    if 'flat_features' in inp:
        flat_features = inp['flat_features']
        if flat_mlp_kwargs is not None:
            flat_features = mlp_helper(flat_features, **flat_mlp_kwargs)
        x = Concatenate()([x, flat_features])
    if final_mlp_kwargs is not None:
        x = mlp_helper(x, **final_mlp_kwargs)

    output_layers = []
    for name in train.target_columns:
        y = x
        output_layers.append(
            Dense(
                units=1,
                activation='sigmoid' if name == 'sex' else None,
                kernel_initializer='he_normal',
                name=name
            )(y)
        )

    if len(output_layers) > 1:
        output = output_layers
    else:
        output = output_layers[0]

    model = keras.Model(inp, output)
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


def _make_input(shape):
    if isinstance(shape, dict):
        return {key: _make_input(value) for key, value in shape.items()}
    else:
        return Input(shape=shape)
