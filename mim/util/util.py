import glob
from itertools import islice

import numpy as np
import pandas as pd


def ranksort(unsorted, ascending=True):
    """
    Returns the rank of each element of an unsorted list. The rank in this
    context means the position of each element had it been sorted.

    Examples:
    ranksort([1, 2, 3, 4]) -> [0, 1, 2, 3]
    ranksort([2, 1, 3, 4]) -> [1, 0, 2, 3]
    ranksort([5, 1, 0, 3]) -> [3, 1, 0, 2]

    :param unsorted: list of comparable elements
    :param ascending: True if rank 0 means the smallest element, False if
    rank 0 means the largest element.
    :return: List of ranks.
    """
    ranks = np.zeros(len(unsorted))
    ranks[np.argsort(unsorted)] = range(len(unsorted))
    if not ascending:
        ranks = len(unsorted) - ranks - 1

    return list(ranks.astype(int))


def public_methods(obj):
    """
    Iterator over all public methods of obj: (method, method_name)-tuple.
    An attribute of obj is considered a public method if it's callable and
    doesn't start with _.

    :param obj: The object who's methods we want
    :return: Iterator over all methods, as a (method, method_name)-tuple.
    """
    for name in dir(obj):
        if callable(getattr(obj, name)) and not name.startswith('_'):
            yield getattr(obj, name), name


def extract_first_float_from_string_series(s):
    """
    Extracts a series of float numbers from a series from strings
    :param s: pandas Series containing strings that contain a number
    :return: pandas Series containing the float numbers,
    result of extracting a number from a string
    """
    extract_float_pattern = r'([0-9]{1,}[.,]?[0-9]{0,})'
    return (
        s.str.extract(extract_float_pattern, expand=False).
        str.replace(",", ".").apply(float))


def hold_out_split(df, size, random_seed=4321):
    """
    Splits the input data into two parts, using the specified random seed.

    :param df: Input data used for split. Index must be unique.
    :param size: Fraction of the rows that should be held-out.
    :param random_seed: Random seed for numpy. Will not be shared with the
    rest of the system.
    :return: Two dataframes as a tuple. The first is the held-out set,
    the second is the remainder.
    """

    random = np.random.RandomState(random_seed)
    df = df.sort_index()
    held_out = random.choice(df.index, replace=False, size=int(size*len(df)))
    rest = set(df.index) - set(held_out)
    return df.loc[held_out, :], df.loc[rest, :]


def keras_model_summary_as_string(model):
    """
    Given a keras model, return the keras.summary() output as a string. If
    you just call model.summary(), the summary will be printed to stdout, and
    the method itself returns None. But if you want to save the output or use
    it in any other way, you need to be fancy about it, apparently. That's
    where this function helper comes in.

    Credit to https://stackoverflow.com/a/53668338/5693369

    :param model: a keras model
    :return: the string that would have been printed if you did
    model.summary()
    """
    result = []
    model.summary(print_fn=lambda x: result.append(x))
    return "\n".join(result)


def callable_to_string(x):
    """
    Returns a reasonable string version of the input if it's a callable.
    Works recursively, so that if the input is a dict or list, all the items
    are checked for callables to stringify.

    A callable is converted to it's module-name followed by it's name, e.g.
    this function is converted to "mim.util.util.callable_to_string"

    :param x: Anything
    :return: Same as the input, except any callables are "stringified" in
    a nested fashion.
    """
    if callable(x):
        return f"{x.__module__}.{x.__name__}"
    elif isinstance(x, list):
        return [callable_to_string(item) for item in x]
    elif isinstance(x, dict):
        return {k: callable_to_string(v) for k, v in x.items()}
    else:
        return x


def insensitive_iglob(pattern, recursive=True):
    """
    Case insensitive version of iglob. Modified slightly from
    stack overflow: https://stackoverflow.com/a/10886685/5693369
    """
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c

    return glob.iglob(''.join(map(either, pattern)), recursive=recursive)


def infer_categorical(data):
    df = pd.DataFrame(data)
    cat, num = [], []
    for col in df.columns:
        if is_categorical(df.loc[:, col]):
            cat.append(col)
        else:
            num.append(col)

    return cat, num


def is_categorical(s):
    return len(s.value_counts()) <= 10


def interpolate(first, last, length):
    """Integer interpolation, rounds all numbers to nearest integer."""
    return list(map(round, np.linspace(first, last, length)))


def take(n, iterable):
    """Return first n items of the iterable as a list. If n is negative,
    return all items.
    """
    if n < 0:
        return list(iterable)

    return list(islice(iterable, n))
