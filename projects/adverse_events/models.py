from keras import layers, losses, optimizers
from keras.models import Model
from keras import backend as K
import tensorflow as tf


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


def anomaly_detector():
    ae = load_trained_model(...)
    new_model = ...

