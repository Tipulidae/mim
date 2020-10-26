from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv1D,
    MaxPool1D,
    Dropout,
)


class BasicCNN(keras.Sequential):
    def __init__(self):
        super().__init__()
        self.add(Input(shape=(1000, 12)))
        self.add_conv_layer()
        self.add_conv_layer()
        self.add(Flatten())
        self.add(Dense(1, activation="sigmoid", kernel_regularizer="l2"))

    def add_conv_layer(self):
        self.add(
            Conv1D(
                filters=32,
                kernel_size=16,
                kernel_regularizer="l2",
                activation="relu",
            )
        )
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

    def compile(self, **kwargs):
        super().compile(
            optimizer='rmsprop',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            **kwargs
        )
