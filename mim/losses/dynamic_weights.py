import sys

import tensorflow as tf

from tensorflow import keras
from keras.losses import LossFunctionWrapper


@keras.utils.register_keras_serializable('dynamic_weights')
class MyBinaryCrossentropy(LossFunctionWrapper):
    def __init__(
        self,
        axis=-1,
        name="my_binary_crossentropy",
        reduction='auto',
    ):
        super().__init__(
            binary_crossentropy,
            name=name,
            reduction=reduction,
            axis=axis,
        )


@keras.utils.register_keras_serializable('dynamic_weights')
class EdenLoss(LossFunctionWrapper):
    def __init__(
        self,
        name="eden_loss_v1",
        reduction='auto',
        target_tpr=0.95,
        tpr_weight=1.0,
    ):
        super().__init__(
            eden_loss_v1,
            name=name,
            reduction=reduction,
            target_tpr=target_tpr,
            tpr_weight=tpr_weight,
        )


@keras.utils.register_keras_serializable('dynamic_weights')
class EdenLossV2(LossFunctionWrapper):
    def __init__(
        self,
        name="eden_loss_v2",
        reduction='auto',
        target_tpr=0.95,
        tpr_weight=1.0,
    ):
        super().__init__(
            eden_loss_v2,
            name=name,
            reduction=reduction,
            target_tpr=target_tpr,
            tpr_weight=tpr_weight,
        )


@keras.utils.register_keras_serializable('dynamic_weights')
class EdenLossV3(LossFunctionWrapper):
    def __init__(
        self,
        name="eden_loss_v3",
        reduction='auto',
        target_tpr=0.95,
        alpha=10.0,
        beta=3.0,
    ):
        super().__init__(
            eden_loss_v3,
            name=name,
            reduction=reduction,
            target_tpr=target_tpr,
            alpha=alpha,
            beta=beta
        )


def eden_loss_v1(y_true, y_pred, target_tpr=0.95, tpr_weight=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    mask = tf.cast(y_true, tf.bool)
    R_plus = tf.boolean_mask(y_pred, mask)
    R_minus = tf.boolean_mask(y_pred, tf.math.logical_not(mask))

    F = tf.math.reduce_mean(R_plus)
    L = tf.math.reduce_mean(R_minus)

    L_F = (L + tpr_weight * tf.math.maximum(0.0, target_tpr - F)) / \
          (1 + tpr_weight)

    tf.print("F: ", F, "L: ", L, "L_F: ", L_F, output_stream=sys.stdout)

    return L_F


def eden_loss_v2(y_true, y_pred, target_tpr=0.95, tpr_weight=1.0):
    # Maybe if the loss is the usual BCE, plus our TPR estimate?
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    mask = tf.cast(y_true, tf.bool)
    R_plus = tf.boolean_mask(y_pred, mask)

    F = tf.math.reduce_mean(R_plus)
    F = tf.math.maximum(0.0, target_tpr - F)
    bce = binary_crossentropy(y_true, y_pred, axis=None)
    loss = (bce + tpr_weight * F) / (1 + tpr_weight)

    tf.print("F: ", F, "\tBCE: ", bce, "\tloss: ", loss,
             output_stream=sys.stdout)
    return loss


def eden_loss_v3(y_true, y_pred, target_tpr=0.95, alpha=10.0, beta=3.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    mask = tf.cast(y_true, tf.bool)
    R_plus = tf.boolean_mask(y_pred, mask)
    R_minus = tf.boolean_mask(y_pred, tf.math.logical_not(mask))

    tpr = tf.math.reduce_mean(R_plus)
    fpr = tf.math.reduce_mean(R_minus)
    constraint = (
        (alpha - beta) * tf.math.maximum(0.0, target_tpr - tpr) +
        beta * tf.math.maximum(0.0, 1 - tpr)
    )

    loss = (fpr + constraint) / (1 + alpha)
    tf.print("Constraint: ", constraint,
             "\tTPR: ", tpr,
             "\tFPR: ", fpr,
             "\tloss: ", loss,
             output_stream=sys.stdout)
    return loss


def binary_crossentropy(y_true, y_pred, axis=-1):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    # Compute cross entropy from probabilities.
    bce = y_true * tf.math.log(y_pred)
    bce += (1 - y_true) * tf.math.log(1 - y_pred)
    return keras.backend.mean(-bce, axis=axis)


class CustomBCE(keras.losses.Loss):
    def __init__(self, regularization_factor=1.0, name="custom_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)

        bce = y_true * tf.math.log(y_pred)
        bce += (1 - y_true) * tf.math.log(1 - y_pred)
        return -bce * self.regularization_factor

    def get_config(self):
        config = {
            'regularization_factor': self.regularization_factor,
        }
        base_config = super().get_config()
        return {**base_config, **config}
