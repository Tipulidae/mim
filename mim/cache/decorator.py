import hashlib
import inspect
import os
import pickle
from itertools import starmap
from types import FunctionType

import pandas as pd

from mim.util.logs import get_logger
from mim.config import PATH_TO_CACHE
from mim.util.metadata import Metadata, MetadataConsistencyException
from mim.cache import settings

log = get_logger('Cache')

# TODO:
# Caching will run into conflicts when several functions with the same name
# are cached, for example if foo.bar and bar.bar are both cached. Method
# names are also problematic if two different classes in different modules
# have the same names and the same cached methods.
#
# Function arguments are now (2018-10-19) considered during caching, so that
# calling foo('bar') is cached separately from foo('baz').
# Known caveats:
# 1. Methods and functions are treated slightly differently. A method is
#  distinguished by containing the argument 'self'. Things will go wrong with
#  methods that lack the argument self, or functions with the argument self.
# 2. Object arguments need a good __repr__ implementation to avoid converting
#  to a string that points to a memory position, such as
#  "<__main__.Something object at 0x7f8577abdda0>"
# 3. Arguments are separated with comma and space:
#  Thus, foo('bar', ', baz'), foo('bar,', ' baz') and foo('bar, ', 'baz')
#  will all be treated as the same function call.


def cache(f):
    """
    Decorator that will save (pickle) the output of a function and return
    the saved output in subsequent calls, rather than re-computing
    everything again. Caching can be toggled on or off for different groups
    of functions. What group a particular function belongs to is decided by
    the input cache_type, and whether that group is cached is determined by
    the corresponding parameter in CacheSettings.

    Apart from the function output, a Metadata report is also saved. On
    subsequent calls, a new Metadata report is created and checked against
    the saved report. Depending on the settings, old cache files may
    be rejected, forcing a re-computed cache. Settings also allow to force
    re-computation if the commit is different, if there are uncommitted
    changes, if the branch is different, or if any of the underlying
    csv-data files are different. Caching can also be bypassed completely
    in the settings.

    The cache file of a class method is named with the string
    representation of the class preceding the function name, like so:
    Foo(1, 2).bar(3, 4, spam=ham).

    Note that methods are differentiated from functions by the first
    argument, which is assumed to be 'self'. If for some reason you have a
    method where the argument is not self, or if you have a regular
    function with a self argument, then things will break!

    :param cache_type: String that determines the category of the function to
    be cached. Whether cache is active for that category is decided by the
    corresponding CacheSettings parameter.
    :return: function that takes a function f and returns a new function f'
    that performs caching.
    """
    def wrapper(*args, **kwargs):
        if settings.enabled:
            # This is a bit of a hack, but it's one of the few ways
            # to check if f is a class method or a function. Note that
            # using inspect.ismethod or isinstance(f, types.MethodType)
            # will both fail, because f is a function in the scope of
            # the class when it is defined...
            is_method = 'self' in inspect.signature(f).parameters
            return _cache(f, args, kwargs, is_method=is_method)
        else:
            log.debug(f'Ignoring cache of {f.__name__}(...)')
            return f(*args, **kwargs)

    return wrapper


def _cache(f, args, kwargs, is_method=False):
    path = function_to_cache_path(f, args, kwargs, is_method=is_method)
    current_metadata = Metadata().report(conda=False)
    if os.path.isfile(path):
        log.debug(f'Loading cached result for {os.path.basename(path)}.')

        with open(path, 'rb') as file:
            cached_metadata = pickle.load(file)
            try:
                settings.validator.validate_consistency(
                    [current_metadata, cached_metadata])
                cached_result = pickle.load(file)
                log.debug('Cached file successfully loaded!')
                return cached_result
            except MetadataConsistencyException as e:
                log.debug(f'Metadata for {os.path.basename(path)} '
                          f'inconsistent, re-computing. {e}')
    else:
        log.debug(f'Cache file {os.path.basename(path)} '
                  f'not found, re-computing!')

    results = f(*args, **kwargs)
    with open(path, 'wb') as file:
        pickle.dump(current_metadata, file)
        pickle.dump(results, file)
    return results


def function_to_cache_path(f, args, kwargs, is_method=False):
    if is_method:
        args_and_kwargs = args_and_kwargs_to_string(*args[1:], **kwargs)
        cache_name = f'{args[0]}.{f.__name__}({args_and_kwargs}).cache'
    else:
        args_and_kwargs = args_and_kwargs_to_string(*args, **kwargs)
        cache_name = f'{f.__qualname__}({args_and_kwargs}).cache'

    return os.path.abspath(
        os.path.join(PATH_TO_CACHE, cache_name))


def args_and_kwargs_to_string(*args, __max_length__=100, **kwargs):
    """
    Will turn input arguments and keyword-arguments into a string
    representation. Pandas objects (index, DataFrame and Series) are turned
    into a SHA1-hash representation of that object. Arguments are separated
    by comma, and keyword-arguments are shown as 'keyword=argument'. If the
    entire string is longer than __max_length__, the SHA1-representation of
    that string is returned instead (40 characters long).

    :param args: n-tuple of arguments to consider.
    :param __max_length__: Max length of the returned string. If the string
    is longer than this, the 40 character SHA1 hash-representation of that
    string is used instead.
    :param kwargs: Keyword arguments to consider.
    :return: String representation of the input args and kwargs.
    """
    args_str = _args_to_string(args)
    kwargs_str = _kwargs_to_string(kwargs)
    result = ', '.join(filter(lambda x: x, [args_str, kwargs_str]))
    if len(result) > __max_length__:
        result = hashlib.sha1(
            result.encode('utf-8'),
            usedforsecurity=False
        ).hexdigest()

    return result


def _args_to_string(args):
    return ', '.join(map(_arg_to_str, args))


def _kwargs_to_string(kwargs):
    return ', '.join(starmap(_kwarg_to_str, kwargs.items()))


def _kwarg_to_str(kw, arg):
    return f'{kw}={_arg_to_str(arg)}'


def _arg_to_str(arg):
    if isinstance(arg, FunctionType):
        return arg.__qualname__
    elif isinstance(arg, (pd.DataFrame, pd.Series, pd.Index)):
        return _hash_pandas(arg)
    else:
        return str(arg)


def _hash_pandas(df):
    hashed_values = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha1(hashed_values, usedforsecurity=False).hexdigest()
