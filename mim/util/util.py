import numpy as np


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
