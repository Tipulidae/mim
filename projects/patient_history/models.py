from tensorflow import keras
from tensorflow.keras.layers import Input, Concatenate, BatchNormalization, \
    Dense, LSTM, Dropout

from mim.models.util import ffnn_helper


def simple_ffnn(train, validation=None, **ffnn_kwargs):
    shapes = train.feature_tensor_shape
    print(f"{shapes=}")
    if isinstance(shapes, dict):
        inp = {
            features: Input(shape=shapes[features])
            for features in shapes
        }
        x = Concatenate()(inp.values())
    else:
        inp = Input(shape=shapes)
        x = inp

    x = BatchNormalization()(x)

    if ffnn_kwargs:
        x = ffnn_helper(x, **ffnn_kwargs)

    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)


def simple_lstm(train, validation=None, **kwargs):
    inp = Input(shape=train.feature_tensor_shape, ragged=True)

    x = LSTM(11)(inp)
    x = Dense(10, activation='relu')(x)
    x = Dropout(0.2)(x)

    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)
