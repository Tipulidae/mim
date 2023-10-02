from tensorflow import keras
from keras.layers import Input, Dense

from mim.models.util import cnn_helper, mlp_helper


def cnn(
        train,
        validation=None,
        cnn_kwargs=None,
        ffnn_kwargs=None,
        final_ffnn_kwargs=None
):
    if final_ffnn_kwargs is None:
        final_ffnn_kwargs = {}

    inp = Input(shape=train.feature_tensor_shape)
    x = cnn_helper(inp, **cnn_kwargs)
    if ffnn_kwargs:
        x = mlp_helper(x, **ffnn_kwargs)
    output = Dense(units=1, activation='sigmoid', kernel_regularizer='l2')(x)

    return keras.Model(inp, output)
