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


def basic_cnn(input_shape, num_conv_layers=2):
    inp = {key: Input(shape=value) for key, value in input_shape.items()}
    layers = []
    if 'ecg' in inp:
        layers.append(_ecg_network(inp['ecg'], num_conv_layers))
    if 'old_ecg' in inp:
        layers.append(_ecg_network(inp['old_ecg'], num_conv_layers))
    if 'features' in inp:
        layers.append(BatchNormalization()(inp['features']))

    if len(layers) > 1:
        x = Concatenate()(layers)
    else:
        x = layers[0]

    output = Dense(1, activation="sigmoid", kernel_regularizer="l2")(x)
    return keras.Model(inp, output)


def _ecg_network(ecg, num_conv_layers, dropout=0.2):
    ecg = BatchNormalization()(ecg)
    for _ in range(num_conv_layers):
        ecg = Conv1D(filters=32, kernel_size=16, kernel_regularizer="l2")(ecg)
        ecg = BatchNormalization()(ecg)
        ecg = ReLU()(ecg)
        ecg = MaxPool1D(pool_size=16)(ecg)
        ecg = Dropout(dropout)(ecg)

    return Flatten()(ecg)


class BasicCNN(keras.Sequential):
    def __init__(self, input_shape=(1200, 8), num_conv_layers=2):
        super().__init__()
        self.add(Input(shape=input_shape))
        self.add(BatchNormalization())
        for _ in range(num_conv_layers):
            self.add_conv_layer()

        self.add(Flatten())
        self.add(Dense(1, activation="sigmoid", kernel_regularizer="l2"))

    def add_conv_layer(self):
        self.add(
            Conv1D(
                filters=32,
                kernel_size=16,
                kernel_regularizer="l2",
            )
        )
        self.add(BatchNormalization())
        self.add(ReLU())
        self.add(MaxPool1D(pool_size=16))
        self.add(Dropout(0.2))

    def compile(self, **kwargs):
        super().compile(
            optimizer='sgd',
            loss='binary_crossentropy',
            metrics=['accuracy'],
            **kwargs
        )


class BasicFF(keras.Sequential):
    def __init__(self):
        super().__init__(
            layers=[
                Flatten(input_shape=(128, ), name='digits'),
                Dense(32, activation='relu', name='hidden'),
                Dense(10, activation='softmax', name='predictions')
            ]
        )
