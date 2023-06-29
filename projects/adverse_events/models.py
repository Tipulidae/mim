import numpy as np
from keras import layers, losses, optimizers
from keras.layers import Concatenate
from keras.models import Model

from mim.models.util import mlp_helper


def _make_input(shape):
    if isinstance(shape, dict):
        return {key: _make_input(value) for key, value in shape.items()}
    else:
        return layers.Input(shape=shape)


def autoencoder_functional(
        train,
        validation=None,
        mlp_kwargs=None, **kwargs
):
    inp = _make_input(train.feature_tensor_shape)

    x = inp
    if mlp_kwargs:
        x = mlp_helper(x, **mlp_kwargs)

    med_output = layers.Dense(units=train.feature_tensor_shape.as_list()[0]-3, activation="sigmoid", name='Med')(x)
    age_output = layers.Dense(1, activation="sigmoid", name='Age')(x)
    gender_output = layers.Dense(2, activation="softmax", name='Gender')(x)
    return Model(inp, {'Med': med_output, 'Age': age_output, 'Gender': gender_output})


def mlp_prediction(train, validation=None, mlp_kwargs=None):
    inp = _make_input(train.feature_tensor_shape)

    x = inp
    if mlp_kwargs:
        x = mlp_helper(x, **mlp_kwargs)

    output = layers.Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return Model(inp, output)


def random_forest():
    return 13
