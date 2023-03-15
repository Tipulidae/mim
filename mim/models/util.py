import math
from typing import Union, List

from keras.layers import (
    AveragePooling1D, Conv1D, BatchNormalization, ReLU, MaxPooling1D,
    Dropout, Flatten, Dense
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
