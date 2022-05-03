from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Concatenate, Dense, BatchNormalization
)

from mim.models.util import ffnn_helper


def simple_ffnn(train, validation=None, **ffnn_kwargs):
    shapes = train['x'].shape
    inp = {
        features: Input(shape=shapes[features])
        for features in shapes
    }
    x = Concatenate()(inp.values())
    x = BatchNormalization()(x)
    x = ffnn_helper(x, **ffnn_kwargs)

    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)
