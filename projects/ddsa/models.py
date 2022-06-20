from tensorflow import keras
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense

from mim.models.util import cnn_helper, ffnn_helper


def ptbxl_cnn(
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
    x = ffnn_helper(x, **ffnn_kwargs)

    output_layers = []

    for name in train.target_columns:
        y = x
        if name in final_ffnn_kwargs:
            y = ffnn_helper(x, **final_ffnn_kwargs[name])

        output_layers.append(
            Dense(
                units=1,
                activation='sigmoid' if name == 'sex' else None,
                kernel_regularizer='l2',
                name=name
            )(y)
        )

    if len(output_layers) > 1:
        output = output_layers
    else:
        output = output_layers[0]

    return keras.Model(inp, output)
