import numpy as np

import tensorflow as tf
from keras.engine import data_adapter
from keras.layers import Input, Dense
from keras.models import Model

from mim.util.logs import get_logger

log = get_logger('Rule out models')


def single_layer_perceptron(train, validation=None):
    inp = Input(shape=train.feature_tensor_shape)
    output = Dense(1, activation='sigmoid', kernel_regularizer=None)(inp)
    return Model(inp, output)
    # model = LossModel(inp, output, sample_weights_shape=len(train.data['y']))
    # return model


class LossModel(Model):
    def __init__(self, *args, sample_weights_shape=2000, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = tf.Variable(np.ones((sample_weights_shape,)))

    def train_step(self, data):
        x, y, index = data_adapter.unpack_x_y_sample_weight(data)
        # log.debug(f"{self.sample_weights=}")
        # index = list(index.as_numpy())
        log.debug(f"{index.dtype=}")
        log.debug(f"{index.shape=}")
        log.debug(f"{tf.squeeze(index).shape=}")
        # new_index = tf.reshape(index, [-1])
        # log.debug(f"{new_index.shape=}")
        # log.debug(f"{type(new_index)=}")
        # log.debug(f"{self.sample_weights.shape=}")
        return super().train_step(
            # (x, y, tf.gather_nd(self.sample_weights, new_index))
            (x, y, tf.gather(self.sample_weights, tf.squeeze(index)))
        )
