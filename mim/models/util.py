import math
from typing import Union, List

import tensorflow as tf
from keras.saving import register_keras_serializable
from keras.layers import (
    AveragePooling1D, Conv1D, BatchNormalization, ReLU, MaxPooling1D,
    Dropout, Flatten, Dense, Input, Add, Activation, GlobalAveragePooling1D,
    multiply, Reshape
)
from keras.regularizers import l2
from keras.optimizers.schedules import LearningRateSchedule

from mim.util.util import interpolate


def cnn_helper(
        x,
        initial_batch_norm=False,
        down_sample=False,
        num_layers=2,
        dropout=0.3,
        dropouts=None,
        filter_first=16,
        filter_last=16,
        filters=None,
        kernel_first=5,
        kernel_last=5,
        kernels=None,
        batch_norm=True,
        batch_norms=None,
        weight_decay=None,
        weight_decays=None,
        pool_size=None,
        pool_sizes=None):

    if down_sample:
        x = AveragePooling1D(2, padding='same')(x)

    if initial_batch_norm:
        x = BatchNormalization()(x)

    if pool_sizes is None:
        if pool_size is None:
            pool_size = _calculate_appropriate_pool_size(
                input_size=x.shape[1],
                num_pools=num_layers,
                minimum_output_size=4
            )
        pool_sizes = num_layers * [pool_size]

    if filters is None:
        if filter_first is not None and filter_last is not None:
            filters = interpolate(filter_first, filter_last, num_layers)
        else:
            raise ValueError('Must specify either filters or both '
                             'filter_first and filter_last. ')

    if kernels is None:
        if kernel_first is not None and kernel_last is not None:
            kernels = interpolate(kernel_first, kernel_last, num_layers)
        else:
            raise ValueError('Must specify either kernels or both '
                             'kernel_first and kernel_last. ')

    if dropouts is None:
        dropouts = num_layers * [dropout]

    if weight_decays is None:
        if weight_decay is None:
            weight_decay = 0.01
        weight_decays = num_layers * [weight_decay]

    if batch_norms is None:
        batch_norms = num_layers * [batch_norm]

    if not _all_lists_have_same_length(
        [filters, kernels, weight_decays, batch_norms, pool_sizes, dropouts],
        expected_length=num_layers
    ):
        raise ValueError("Not all parameters have the same length!")

    for layer in range(num_layers):
        x = Conv1D(
            filters=filters[layer],
            kernel_size=kernels[layer],
            kernel_regularizer=l2(weight_decays[layer]),
            padding='same')(x)
        if batch_norms[layer]:
            x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling1D(pool_size=pool_sizes[layer])(x)
        x = Dropout(dropouts[layer])(x)

    x = Flatten()(x)
    return x


def mlp_helper(
        x, sizes,
        activation='relu',
        dropout: Union[float, List[float]] = 0.0,
        batch_norm: Union[bool, List[bool]] = False,
        regularizer: Union[float, List[float], dict] = 0.0
):
    """

    :param x:
    :param sizes:
    :param activation:
    :param dropout:
    :param batch_norm:
    :param regularizer: Specify the l2 regularization weights. If it's a
    float, uses the same weight for kernel, bias and activation, for all
    layers. Use a list to specify different weights for each layer. Use a
    dict to specify different weights for kernel, bias and activation
    regularization.
    :return:
    """
    num_layers = len(sizes)

    def parse_batch_norm(arg):
        if isinstance(arg, bool):
            return num_layers * [arg]
        elif isinstance(arg, list):
            assert len(arg) == num_layers
            return arg
        else:
            raise TypeError

    def parse_dropout(arg):
        if isinstance(arg, float):
            return num_layers * [arg]
        elif isinstance(arg, list):
            assert len(arg) == num_layers
            return arg
        else:
            raise TypeError

    def parse_regularizers(arg):
        if isinstance(arg, dict):
            assert {'activity', 'kernel', 'bias'}.issubset(arg.keys())
            return (
                parse_regularizer(arg['activity']),
                parse_regularizer(arg['kernel']),
                parse_regularizer(arg['bias'])
            )
        else:
            return 3 * (parse_regularizer(arg),)

    def parse_regularizer(arg):
        if isinstance(arg, float):
            return num_layers * [arg]
        elif isinstance(arg, list):
            assert len(arg) == num_layers
            return arg
        else:
            raise TypeError

    activity, kernel, bias = parse_regularizers(regularizer)
    dropouts = parse_dropout(dropout)
    batch_norms = parse_batch_norm(batch_norm)
    assert _all_lists_have_same_length(
        [sizes, dropouts, batch_norms, activity, kernel, bias]
    )

    for layer in range(num_layers):
        x = Dense(
            sizes[layer],
            activation=activation,
            activity_regularizer=l2(activity[layer]),
            kernel_regularizer=l2(kernel[layer]),
            bias_regularizer=l2(bias[layer])
        )(x)

        if batch_norms[layer]:
            x = BatchNormalization()(x)
        if dropouts[layer] > 0:
            x = Dropout(dropouts[layer])(x)

    return x


def _all_lists_have_same_length(lists, expected_length=None):
    if expected_length is None:
        expected_length = len(lists[0])

    return all(map(lambda x: len(x) == expected_length, lists))


def _calculate_appropriate_pool_size(
        input_size, num_pools, minimum_output_size=4):
    """
    Calculate what the pool size should be if we start with input_size and
    pool num_pools times, and want to end up with a size that is at least
    minimum_output_size.
    :param input_size:
    :param num_pools:
    :param minimum_output_size:
    :return:
    """
    return math.floor((input_size / minimum_output_size) ** (1 / num_pools))


def make_input(shape):
    if isinstance(shape, dict):
        return {key: make_input(value) for key, value in shape.items()}
    else:
        return Input(shape=shape)


class ResidualUnit(object):
    """Residual unit block (unidimensional).
    Parameters
    ----------
    n_samples_out: int
        Number of output samples.
    n_filters_out: int
        Number of output filters.
    kernel_initializer: str, otional
        Initializer for the weights matrices. See Keras initializers. By
        default it uses 'he_normal'.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. Default is 17.
    preactivation: bool, optional
        When preactivation is true use full preactivation architecture proposed
        in [1]. Otherwise, use architecture proposed in the original ResNet
        paper [2]. By default it is true.
    postactivation_bn: bool, optional
        Defines if you use batch normalization before or after the activation
        layer (there seems to be some advantages in some cases:
        https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md).
        If true, the batch normalization is used before the activation
        function, otherwise the activation comes first, as it is usually done.
        By default it is false.
    activation_function: string, optional
        Keras activation function to be used. By default 'relu'.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep
           Residual Networks," arXiv:1603.05027 [cs], Mar. 2016.
           https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for
           Image Recognition," in 2016 IEEE Conference on Computer Vision
           and Pattern Recognition (CVPR), 2016, pp. 770-778.
           https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, n_samples_out, n_filters_out,
                 kernel_initializer='he_normal',
                 dropout_rate=0.8, kernel_size=17, preactivation=True,
                 postactivation_bn=False, activation_function='relu',
                 use_se_layer=False, reduction_ratio=8,
                 ):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function
        self.use_se_layer = use_se_layer
        self.reduction_ratio = reduction_ratio

    def _skip_connection(self, y, downsample, n_filters_in):
        """Implement skip connection."""
        # Deal with downsampling
        if downsample > 1:
            y = MaxPooling1D(downsample, strides=downsample, padding='same')(y)
        elif downsample == 1:
            y = y
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            y = Conv1D(self.n_filters_out, 1, padding='same',
                       use_bias=False,
                       kernel_initializer=self.kernel_initializer)(y)
        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = Activation(self.activation_function)(x)
            x = BatchNormalization(center=False, scale=False)(x)
        else:
            x = BatchNormalization()(x)
            x = Activation(self.activation_function)(x)
        return x

    def _squeeze_excite_block(self, input):
        x = GlobalAveragePooling1D()(input)
        x = Reshape((1, self.n_filters_out))(x)
        x = Dense(self.n_filters_out // self.reduction_ratio,
                  activation='relu')(x)
        x = Dense(self.n_filters_out, activation='sigmoid')(x)
        x = multiply([input, x])
        return x

    def __call__(self, inputs):
        """Residual unit."""
        x, y = inputs
        n_samples_in = y.shape[1]
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[2]
        y = self._skip_connection(y, downsample, n_filters_in)
        # 1st layer
        x = Conv1D(
            self.n_filters_out,
            self.kernel_size,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer
        )(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)

        # 2nd layer
        x = Conv1D(
            self.n_filters_out,
            self.kernel_size,
            strides=downsample,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer
        )(x)

        if self.use_se_layer:
            x = self._squeeze_excite_block(x)

        if self.preactivation:
            x = Add()([x, y])  # Sum skip connection and main connection
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        else:
            x = BatchNormalization()(x)
            x = Add()([x, y])  # Sum skip connection and main connection
            x = Activation(self.activation_function)(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            y = x
        return [x, y]


class ResidualUnitV2:
    """
    Trying to re-create the architecture described in
    10.1038/s41598-022-24254-x, which is similar but not identical to the
    resnet described above. Biggest differences is that this version has
    the batch-norm, relu and dropout before the conv-layer, and after two
    convolutions there's a "squeeze and excite" layer.
    This architecture also uses a weight-decay on the convolutional layers.
    """
    def __init__(self,
                 n_samples_out,
                 n_filters_out,
                 kernel_initializer='he_normal',
                 dropout_rate=0.5,
                 kernel_size=17,
                 reduction_ratio=8,
                 activation_function='relu'):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.activation_function = activation_function
        self.reduction_ratio = reduction_ratio

    def _skip_connection(self, y, downsample, n_filters_in):
        """Implement skip connection."""
        # Deal with downsampling
        if downsample > 1:
            y = MaxPooling1D(downsample, strides=downsample, padding='same')(y)
        elif downsample == 1:
            y = y
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            y = Conv1D(self.n_filters_out, 1, padding='same',
                       use_bias=False,
                       kernel_initializer=self.kernel_initializer)(y)
        return y

    def _batch_norm_plus_activation(self, x):
        x = BatchNormalization()(x)
        x = Activation(self.activation_function)(x)
        return x

    def _squeeze_excite_block(self, input):
        x = GlobalAveragePooling1D()(input)
        x = Reshape((1, self.n_filters_out))(x)
        x = Dense(self.n_filters_out // self.reduction_ratio,
                  activation='relu')(x)
        x = Dense(self.n_filters_out, activation='sigmoid')(x)
        x = multiply([input, x])
        return x

    def __call__(self, inputs):
        """Residual unit."""
        # (BN -> Relu -> dropout -> conv)*2 -> SE -> Add -> out
        x, y = inputs
        n_samples_in = y.shape[1]
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[2]
        y = self._skip_connection(y, downsample, n_filters_in)

        # 1st layer
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        x = Conv1D(
            self.n_filters_out,
            self.kernel_size,
            padding='same',
            use_bias=False,
            kernel_regularizer=l2(0.001),
            kernel_initializer=self.kernel_initializer
        )(x)

        # 2nd layer
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        x = Conv1D(
            self.n_filters_out,
            self.kernel_size,
            strides=downsample,
            padding='same',
            use_bias=False,
            kernel_regularizer=l2(0.001),
            kernel_initializer=self.kernel_initializer
        )(x)

        # SE layer here
        x = self._squeeze_excite_block(x)

        # Join with skip-connection
        x = Add()([x, y])  # Sum skip connection and main connection
        y = x

        return [x, y]


# Copy-paste from tensorflow 2.14. Current version doesn't have the warmup
# thing.
@register_keras_serializable(package="mim.models")
class CosineDecayWithWarmup(LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay with optional warmup.

    See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
    SGDR: Stochastic Gradient Descent with Warm Restarts.

    For the idea of a linear warmup of our learning rate,
    see [Goyal et al.](https://arxiv.org/pdf/1706.02677.pdf).

    When we begin training a model, we often want an initial increase in our
    learning rate followed by a decay. If `warmup_target` is an int, this
    schedule applies a linear increase per optimizer step to our learning rate
    from `initial_learning_rate` to `warmup_target` for a duration of
    `warmup_steps`. Afterwards, it applies a cosine decay function taking our
    learning rate from `warmup_target` to `alpha` for a duration of
    `decay_steps`. If `warmup_target` is None we skip warmup and our decay
    will take our learning rate from `initial_learning_rate` to `alpha`.
    It requires a `step` value to  compute the learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.

    The schedule is a 1-arg callable that produces a warmup followed by a
    decayed learning rate when passed the current optimizer step. This can be
    useful for changing the learning rate value across different invocations of
    optimizer functions.

    Our warmup is computed as:

    ```python
    def warmup_learning_rate(step):
        completed_fraction = step / warmup_steps
        total_delta = target_warmup - initial_learning_rate
        return completed_fraction * total_delta
    ```

    And our decay is computed as:

    ```python
    if warmup_target is None:
        initial_decay_lr = initial_learning_rate
    else:
        initial_decay_lr = warmup_target

    def decayed_learning_rate(step):
        step = min(step, decay_steps)
        cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return initial_decay_lr * decayed
    ```

    Example usage without warmup:

    ```python
    decay_steps = 1000
    initial_learning_rate = 0.1
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps)
    ```

    Example usage with warmup:

    ```python
    decay_steps = 1000
    initial_learning_rate = 0
    warmup_steps = 1000
    target_learning_rate = 0.1
    lr_warmup_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps, warmup_target=target_learning_rate,
        warmup_steps=warmup_steps
    )
    ```

    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.

    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """

    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        alpha=0.0,
        name=None,
        warmup_target=None,
        warmup_steps=0,
        decay_epochs=0,
        warmup_epochs=0,
        steps_per_epoch=0
    ):
        """Applies cosine decay to the learning rate.

        Args:
          initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python int. The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python int.
            Number of steps to decay over.
          alpha: A scalar `float32` or `float64` `Tensor` or a Python int.
            Minimum learning rate value for decay as a fraction of
            `warmup_target`.
          name: String. Optional name of the operation.  Defaults to
            'CosineDecay'.
          warmup_target: None or a scalar `float32` or `float64` `Tensor` or a
            Python int. The target learning rate for our warmup phase. Will
            cast to the `initial_learning_rate` datatype. Setting to None will
            skip warmup and begins decay phase from `initial_learning_rate`.
            Otherwise scheduler will warmup from `initial_learning_rate` to
            `warmup_target`.
          warmup_steps: A scalar `int32` or `int64` `Tensor` or a Python int.
            Number of steps to warmup over.
        """
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.alpha = alpha
        self.name = name
        self.warmup_target = warmup_target

        if steps_per_epoch > 0 and decay_epochs > 0:
            self.decay_steps = math.ceil(decay_epochs * steps_per_epoch)
            self.warmup_steps = math.ceil(warmup_epochs * steps_per_epoch)
        else:
            self.decay_steps = decay_steps
            self.warmup_steps = warmup_steps

    def _decay_function(self, step, decay_steps, decay_from_lr, dtype):
        with tf.name_scope(self.name or "CosineDecay"):
            completed_fraction = step / decay_steps
            tf_pi = tf.constant(math.pi, dtype=dtype)
            cosine_decayed = 0.5 * (1.0 + tf.cos(tf_pi * completed_fraction))
            decayed = (1 - self.alpha) * cosine_decayed + self.alpha
            return tf.multiply(decay_from_lr, decayed)

    def _warmup_function(
        self, step, warmup_steps, warmup_target, initial_learning_rate, dtype
    ):
        with tf.name_scope(self.name or "CosineDecay"):
            completed_fraction = step / warmup_steps
            total_step_delta = warmup_target - initial_learning_rate
            total_step_delta = tf.cast(total_step_delta, dtype)
            return (total_step_delta * completed_fraction +
                    initial_learning_rate)

    def __call__(self, step):
        with tf.name_scope(self.name or "CosineDecay"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            global_step_recomp = tf.cast(step, dtype)

            if self.warmup_target is None:
                global_step_recomp = tf.minimum(global_step_recomp,
                                                decay_steps)
                return self._decay_function(
                    global_step_recomp,
                    decay_steps,
                    initial_learning_rate,
                    dtype,
                )

            warmup_target = tf.cast(self.warmup_target, dtype)
            warmup_steps = tf.cast(self.warmup_steps, dtype)

            global_step_recomp = tf.minimum(
                global_step_recomp, decay_steps + warmup_steps
            )

            return tf.cond(
                global_step_recomp < warmup_steps,
                lambda: self._warmup_function(
                    global_step_recomp,
                    warmup_steps,
                    warmup_target,
                    initial_learning_rate,
                    dtype
                ),
                lambda: self._decay_function(
                    global_step_recomp - warmup_steps,
                    decay_steps,
                    warmup_target,
                    dtype,
                ),
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "name": self.name,
            "warmup_target": self.warmup_target,
            "warmup_steps": self.warmup_steps
        }
