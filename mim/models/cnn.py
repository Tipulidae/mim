import tensorflow.keras as keras
from tensorflow.keras import layers


class BasicCNN(keras.Sequential):
    def __init__(self):
        super().__init__()
        self.add(layers.Input(shape=(1000, 12)))
        self.add_conv_layer()
        self.add_conv_layer()
        self.add(layers.Flatten())
        self.add(layers.Dense(
            1, activation="sigmoid", kernel_regularizer="l2"))

    def add_conv_layer(self):
        self.add(
            layers.Conv1D(
                filters=32,
                kernel_size=16,
                kernel_regularizer="l2",
                activation="relu",
            )
        )
        self.add(layers.MaxPool1D(pool_size=16))
        self.add(layers.Dropout(0.2))

    def compile(self, **kwargs):
        super().compile(
            optimizer='sgd',
            loss='binary_crossentropy',
            metrics=['accuracy'],
            **kwargs
        )

    def fit(self, **kwargs):
        return super().fit(
            batch_size=64,
            epochs=5,
            **kwargs
        )
