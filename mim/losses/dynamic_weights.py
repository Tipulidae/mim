import tensorflow as tf

from tensorflow import keras


class CustomBCE(keras.losses.Loss):
    def __init__(self, regularization_factor=1.0, name="custom_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        # y_true = tf.convert_to_tensor(y_true)
        # y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        bce = y_true * tf.math.log(y_pred)
        bce += (1 - y_true) * tf.math.log(1 - y_pred)
        return -bce * self.regularization_factor

        # mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        # reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        # return mse + reg * self.regularization_factor

    def get_config(self):
        config = {
            'regularization_factor': self.regularization_factor,
        }
        base_config = super().get_config()
        return {**base_config, **config}

        # config = super().get_config()
        # config.update({"regularization_factor": self.regularization_factor})
        # return config
