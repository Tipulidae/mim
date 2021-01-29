from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv1D,
    MaxPool1D,
    Dropout,
    BatchNormalization,
    Concatenate,
    ReLU
)


def basic_cnn2(train, validation=None, dropout=0, layers=None,
               hidden_layer=None):
    x = inp = Input(shape=train['x'].shape)
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


def basic_cnn(train, *, num_conv_layers=2, dropout=0.3, filters=32,
              kernel_size=16):
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
