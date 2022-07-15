import math

from tensorflow.keras.layers import (
    AveragePooling1D, Conv1D, BatchNormalization, ReLU, MaxPooling1D,
    Dropout, Flatten, Dense
)
from tensorflow.keras.regularizers import l2

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


def ffnn_helper(
        x, sizes,
        activation='relu',
        dropouts=None,
        default_dropout=0.0,
        batch_norms=None,
        default_regularizer=0.0,
        activity_regularizer=None,
        activity_regularizers=None,
        kernel_regularizer=None,
        kernel_regularizers=None,
        bias_regularizer=None,
        bias_regularizers=None):
    num_layers = len(sizes)
    if dropouts is None:
        dropouts = num_layers * [default_dropout]
    if batch_norms is None:
        batch_norms = num_layers * [False]
    if activity_regularizers is None:
        if activity_regularizer is None:
            activity_regularizer = default_regularizer
        activity_regularizers = num_layers*[activity_regularizer]
    if kernel_regularizers is None:
        if kernel_regularizer is None:
            kernel_regularizer = default_regularizer
        kernel_regularizers = num_layers*[kernel_regularizer]
    if bias_regularizers is None:
        if bias_regularizer is None:
            bias_regularizer = default_regularizer
        bias_regularizers = num_layers*[bias_regularizer]

    assert _all_lists_have_same_length(
        [sizes, dropouts, batch_norms, activity_regularizers,
         kernel_regularizers, bias_regularizers]
    )

    for layer in range(num_layers):
        x = Dense(
            sizes[layer],
            activation=activation,
            activity_regularizer=l2(activity_regularizers[layer]),
            kernel_regularizer=l2(kernel_regularizers[layer]),
            bias_regularizer=l2(bias_regularizers[layer])
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
