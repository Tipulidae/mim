import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Concatenate, LSTM

from mim.models.util import mlp_helper, make_input


def mlp(train, validation=None, patient_mlp_kwargs=None, bw_mlp_kwargs=None,
        final_mlp_kwargs=None):
    inp = make_input(train.feature_tensor_shape)
    layers = []
    if 'patient_features' in inp:
        patient = inp['patient_features']
        if patient_mlp_kwargs is not None:
            layers.append(mlp_helper(patient, **patient_mlp_kwargs))
        else:
            layers.append(patient)

    if 'eeg_features' in inp:
        if bw_mlp_kwargs is not None:
            layers.append(
                mlp_helper(inp['eeg_features'], **bw_mlp_kwargs))
        else:
            layers.append(inp['eeg_features'])

    if len(layers) > 1:
        x = Concatenate()(layers)
    else:
        x = layers[0]

    if final_mlp_kwargs:
        x = mlp_helper(x, **final_mlp_kwargs)

    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)


def lstm(train, validation=None, lstm_kwargs=None, patient_mlp_kwargs=None,
         final_mlp_kwargs=None,):
    # I need this apparently to avoid a "fail to find dnn implementation"
    # error.
    device = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(device, True)

    inp = make_input(train.feature_tensor_shape)
    if lstm_kwargs is None:
        lstm_kwargs = {}

    x = LSTM(**lstm_kwargs)(inp['eeg_features'])

    if 'patient_features' in inp:
        if patient_mlp_kwargs is not None:
            patient = mlp_helper(
                inp['patient_features'],
                **patient_mlp_kwargs
            )
        else:
            patient = inp['patient_features']

        x = Concatenate()([x, patient])

    if final_mlp_kwargs is not None:
        x = mlp_helper(x, **final_mlp_kwargs)

    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)
