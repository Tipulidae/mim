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

"""
class AnomalyDetector(Model):
    def __init__(self, dropout=0.0, latent_dim=60, num_layers=3):
        super(AnomalyDetector, self).__init__()

        factor = (589 / latent_dim) ** (1 / num_layers)
        layer1 = round(latent_dim * factor)
        layer2 = round(latent_dim * factor * factor)

        encoder_input = tf.keras.Input(589)
        drop_out = layers.Dropout(dropout, input_shape=(589,))(encoder_input)
        hidden1 = layers.Dense(layer2, activation="relu")(drop_out)
        hidden2 = layers.Dense(layer1, activation="relu")(hidden1)
        encoder_output = layers.Dense(latent_dim, activation="relu")(hidden2)
        self.encoder = tf.keras.Model(encoder_input, encoder_output, name="encoder")

        decoder_input = tf.keras.Input(shape=(latent_dim,))
        hidden1 = layers.Dense(layer1, activation="relu")(decoder_input)
        hidden2 = layers.Dense(layer2, activation="relu")(hidden1)
        decoder_output1 = layers.Dense(589, activation="relu")(hidden2)

        self.decoder = tf.keras.Model(decoder_input, outputs=decoder_output1, name="decoder")

    def call(self, x, training=None, mask=None):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def autoencoder(train, validation=None, **kwargs):
    return AnomalyDetector(**kwargs)
"""


def autoencoder_functional(
        train,
        validation=None,
        mlp_kwargs=None, **kwargs
):
    inp = _make_input(train.feature_tensor_shape)

    x = inp
    if mlp_kwargs:
        x = mlp_helper(x, **mlp_kwargs)

    # {'x': ..., 'y': {'y1': ..., 'y2': ...}}

    med_output = layers.Dense(units=train.feature_tensor_shape.as_list()[0]-3, activation="sigmoid", name='Med')(x)
    age_output = layers.Dense(1, activation="sigmoid", name='Age')(x)
    gender_output = layers.Dense(2, activation="softmax", name='Gender')(x)
    return Model(inp, {'Med': med_output, 'Age': age_output, 'Gender': gender_output})
