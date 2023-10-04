import math
from typing import Union, List

from keras.layers import (
    AveragePooling1D, Conv1D, BatchNormalization, ReLU, MaxPooling1D,
    Dropout, Flatten, Dense, Input, Add, Activation, GlobalAveragePooling1D,
    multiply, Reshape
)
from keras.regularizers import l2

from mim.util.util import interpolate


def cnn_helper(
        x,
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

    assert _all_lists_have_same_length(
        [filters, kernels, weight_decays, batch_norms, pool_sizes, dropouts],
        expected_length=num_layers
    )

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
                 postactivation_bn=False, activation_function='relu'):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function

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
