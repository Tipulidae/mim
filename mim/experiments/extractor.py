import random
import itertools
from copy import copy
from typing import Dict, TypeVar, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import h5py
from torch.utils.data import Dataset
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from mim.util.util import infer_categorical


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
T_dict = Dict[str, T_co]
T_tuple = Tuple[T_co, ...]
T_stack = TypeVar('T_stack', T_tuple, T_dict)


class Data:
    def __init__(
            self,
            data,
            index=None,
            columns=None,
            dtype=None,
            fits_in_memory=True,
            groups=None,
            predefined_splits=None,
            shape=None,
    ):
        self.data = data
        if columns is None and not isinstance(data, dict):
            num_features = np.shape(data)[-1]
            self._columns = list(range(num_features))
        else:
            self._columns = columns

        if dtype is None:
            self.dtype = infer_dtype(data)
        elif isinstance(dtype, dict):
            self.dtype = dtype
        else:
            self.dtype = tf.as_dtype(dtype)

        self.groups = groups
        self.predefined_splits = predefined_splits
        self._fits_in_memory = fits_in_memory
        if index is None:
            self._index = np.array(range(len(data)))
        else:
            self._index = index

        if shape is None:
            self._shape = infer_shape(data)
        else:
            self._shape = shape

    def split(self, index_a, index_b):
        return self.lazy_slice(index_a), self.lazy_slice(index_b)

    def lazy_slice(self, index):
        new_data = copy(self)
        new_data._index = np.array([self._index[i] for i in index])

        if new_data.groups is not None:
            new_data.groups = [self.groups[i] for i in index]
        if new_data.predefined_splits is not None:
            new_data.predefined_splits = [
                self.predefined_splits[i] for i in index]

        return new_data

    @property
    def index(self):
        return np.array(range(len(self)))

    @property
    def columns(self):
        return self._columns

    def as_dataset(self, shuffle=False, seed=123):
        if shuffle and not self._fits_in_memory:
            r = random.Random(seed)

            def randomize():
                new_index = r.sample(range(len(self)), k=len(self))
                for i in new_index:
                    yield self[i]

            generator = randomize
        else:
            generator = self

        if self._fits_in_memory:
            return tf.data.Dataset.from_tensor_slices(
                generator.as_numpy()
            )
        else:
            return tf.data.Dataset.from_generator(
                generator=generator,
                output_types=self.type,
                output_shapes=self.shape
            )

    def as_numpy(self):
        return np.array(list(self))

    def as_flat_numpy(self):
        """Returns a flattened view of the data. Each item in the underlying
        data structure is flattened.
        """
        x = self.as_numpy()
        if len(x.shape) == 1:
            return x.reshape(-1, 1)
        else:
            return x.reshape(x.shape[0], np.prod(x.shape[1:]))

    @property
    def type(self):
        return self.dtype

    @property
    def shape(self):
        return tf.TensorShape(self._shape)

    @property
    def fits_in_memory(self):
        return self._fits_in_memory

    @fits_in_memory.setter
    def fits_in_memory(self, setting):
        self._fits_in_memory = setting

    def __call__(self, *args, **kwargs):
        return self.__iter__()

    def __getitem__(self, item):
        return self.data[self._index[item]]

    def __len__(self):
        return len(self._index)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def partition(xs):
    """
    Turn an input list of objects into a sequence of indices, one for each
    unique object, such that the index shows the location in the input list
    where that object is. Example:

    [0, 0, 1, 0, 1, 2, 2] -> [0, 1, 3], [2, 4], [5, 6]

    :param xs:
    :return:
    """
    items = set(xs)
    for y in sorted(items):
        yield [i for i, x in enumerate(xs) if x == y]


class RaggedData(Data):
    def __init__(self, *args, slices=None, **kwargs):
        super().__init__(*args, **kwargs)
        if len(slices) != len(self.index):
            raise ValueError("Ragged slices must be a list of same length as "
                             "the index.")
        self.slices = slices
        self._shape = [None] + self._shape

    def as_numpy(self):
        ind = []
        row_lengths = []
        for i in self._index:
            start, stop = self.slices[i]
            ind += list(range(start, stop))
            row_lengths.append(stop - start)

        return tf.RaggedTensor.from_row_lengths(self.data[ind], row_lengths)

    def __getitem__(self, item):
        return self.data[range(*self.slices[self._index[item]])]


class Container(Data):
    def __init__(self, data: Dict[str, Data], index=None, fits_in_memory=None,
                 **kwargs):
        if not isinstance(data, dict):
            raise TypeError(f"Data must be of type dict (was {type(data)})")
        if len({len(v) for v in data.values()}) != 1:
            raise ValueError("Inconsistent length of constituent Data")
        if index is None:
            index = next(iter(data.values())).index
        if fits_in_memory is None:
            fits_in_memory = all([d.fits_in_memory for d in data.values()])

        super().__init__(data, index=index, fits_in_memory=fits_in_memory,
                         **kwargs)
        self.fits_in_memory = fits_in_memory

    def lazy_slice(self, index):
        c = self.__class__(
            {key: value.lazy_slice(index) for key, value in self.data.items()},
            dtype=self.type
        )
        if self.predefined_splits is not None:
            c.predefined_splits = [self.predefined_splits[i] for i in index]
        if self.groups is not None:
            c.groups = [self.groups[i] for i in index]
        return c

    @classmethod
    def from_dict(cls, data_dict):
        def to_data(item):
            if isinstance(item, Data):
                return item
            else:
                return Data(item)

        return cls(
            {key: to_data(value) for key, value in data_dict.items()}
        )

    @property
    def type(self):
        return {key: value.type for key, value in self.data.items()}

    @property
    def columns(self):
        return {key: value.columns for key, value in self.data.items()}

    @property
    def shape(self):
        return {key: value.shape for key, value in self.data.items()}

    def as_numpy(self):
        return {key: value.as_numpy() for key, value in self.data.items()}

    def as_flat_numpy(self):
        return np.concatenate(
            [x.as_flat_numpy() for x in self.data.values()],
            axis=1
        )

    @property
    def fits_in_memory(self):
        return self._fits_in_memory

    @fits_in_memory.setter
    def fits_in_memory(self, setting):
        self._fits_in_memory = setting
        for value in self.data.values():
            value.fits_in_memory = setting

    def __getitem__(self, item):
        if item in self.data:
            return self.data[item]
        else:
            return {key: value[item] for key, value in self.data.items()}


class DataWrapper:
    def __init__(self, features, labels, index, groups=None,
                 predefined_splits=None, fits_in_memory=True):
        """
        Features and labels should either be tuples of data, columns, where
        data is array-like and columns is a list of column-names corresponding
        to the data, OR features and labels are potentially nested dicts of
        such tuples.

        index should be a tuple of data, columns, where data is array-like
        and columns is a list of column-names.

        :param features:
        :param labels:
        :param index:
        :param groups:
        :param predefined_splits:
        :param fits_in_memory:
        """
        self.data = Container(
            {
                'x': features,
                'y': labels,
                'index': index,
            },
            index=range(len(index)),
            groups=groups,
            predefined_splits=predefined_splits,
            fits_in_memory=fits_in_memory
        )

    @property
    def feature_names(self):
        return self.data['x'].columns

    @property
    def target_columns(self):
        return self.data['y'].columns

    @property
    def output_size(self):
        return len(self.data['y'].columns)

    @property
    def index(self):
        return self.data.index

    @property
    def groups(self):
        return self.data.groups

    @property
    def predefined_splits(self):
        return self.data.predefined_splits

    @property
    def feature_tensor_shape(self):
        return self.data['x'].shape

    @property
    def y(self):
        return self.to_dataframe(self.data['y'].as_flat_numpy())

    def __len__(self):
        return len(self.data['index'])

    def x(self, can_use_tf_dataset=True):
        if can_use_tf_dataset:
            return self.data['x'].as_dataset()
        else:
            return self.data['x'].as_flat_numpy()

    def to_dataframe(self, y):
        """
        Turn the input array-like into a pandas DataFrame, where the index
        and columns matches those of the underlying data labels. In other
        words, the output will be a DataFrame with column names corresponding
        to the data output columns (e.g. "mace30"), and the index will match
        the index of the data set (e.g. patient-id).
        :param y:
        :return:
        """
        index_data = self.data['index'].as_numpy()
        if len(index_data.shape) > 1:
            index = pd.MultiIndex.from_arrays(
                list(index_data.T),
                names=self.data['index'].columns
            )
        else:
            index = pd.Index(
                index_data,
                name=self.data['index'].columns[0]
            )

        columns = self.data['y'].columns
        if isinstance(columns, dict):
            columns = list(itertools.chain(*columns.values()))
        return pd.DataFrame(y, index=index, columns=columns)

    def as_dataset(self, batch_size=1, prefetch=0, shuffle=True, **kwargs):
        x = self.data['x'].as_dataset(shuffle=shuffle)
        y = self.data['y'].as_dataset(shuffle=shuffle)
        # index = tf.data.Dataset.from_tensors(list(range(len(y))))
        fixed_data = tf.data.Dataset.zip((x, y))

        # If the data _does_ fit in memory, we can use the tf shuffling
        # instead. This would be bad if data doesn't fit in memory though,
        # because tf will load the entire dataset in memory before shuffling.
        if shuffle and self.data.fits_in_memory:
            fixed_data = fixed_data.shuffle(len(self.data))

        fixed_data = fixed_data.batch(batch_size)

        if prefetch:
            fixed_data = fixed_data.prefetch(prefetch)

        return fixed_data

    def as_dataloader(self, batch_size=1, prefetch=None, shuffle=True,
                      random_seed=123, **kwargs):
        # This will automatically convert any numpy arrays etc to
        # pytorch Tensors
        # Might have to use pin_memory and pin_memory_device to work on the
        # GPU properly.
        # TODO: when torch 2.1+ is working,
        # we can change this to torch.utils.data.StackDataset and get rid of
        # the copy-pasted version here.
        data = StackDataset(
            self.data['x'], self.data['y'].as_flat_numpy(), self.data.index
        )
        if shuffle:
            g = torch.Generator()
            g.manual_seed(random_seed)
        else:
            g = None

        dataloader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=shuffle, generator=g,
            prefetch_factor=prefetch, **kwargs
        )
        return dataloader

    def as_numpy(self):
        x = self.data['x'].as_flat_numpy()
        y = self.data['y'].as_numpy().ravel()
        return x, y

    def as_dataframe(self):
        x = self.data['x'].as_flat_numpy()
        y = self.data['y'].as_numpy()

        index = pd.Index(
            self.data['index'].as_numpy(),
            name=self.data['index'].columns[0]
        )
        x_columns = self.data['x'].columns
        if isinstance(x_columns, dict):
            x_columns = list(itertools.chain(*x_columns.values()))

        y_columns = self.data['y'].columns
        if isinstance(y_columns, dict):
            y_columns = list(itertools.chain(*y_columns.values()))

        x = pd.DataFrame(x, index=index, columns=x_columns)
        y = pd.DataFrame(y, index=index, columns=y_columns)

        return x, y

    def split(self, index_a, index_b):
        return self.lazy_slice(index_a), self.lazy_slice(index_b)

    def lazy_slice(self, index):
        new_data = copy(self)
        new_data.data = new_data.data.lazy_slice(index)
        return new_data

    def lazy_partition(self, xs):
        for p in partition(xs):
            new_data = copy(self)
            new_data.data = new_data.data.lazy_slice(p)
            yield new_data


def infer_shape(data):
    shape = np.shape(data)
    n = len(shape)
    if n == 1:
        return []
    elif n < 1:
        return None

    return list(shape[1:])


def infer_dtype(data):
    if hasattr(data, 'dtype'):
        return tf.as_dtype(data.dtype)
    elif isinstance(data, list):
        return infer_dtype(data[0])
    elif isinstance(data, float):
        return tf.float64
    elif isinstance(data, bool):
        return tf.bool
    else:
        return tf.int64


def sklearn_process(split_number=0, **processors):
    """
    Returns a function that takes a train, val Data tuple as input and
    processes all of the underlying data with the specified processor,
    returning new Data (or Container) objects.

    The training data is used to learn parameters, for scaling etc., and
    applied to the validation data. Categorical columns are identified
    automatically, but the returned Data may change the ordering of the
    columns.

    The processor can be a sklearn Pipeline, or a Transformer like the
    StandardScaler.

    TODO:
    I would like to be able to apply different processors to different
    "branches" of the input features. For instance, to apply PCA only on the
    "Forberg"-features, and not the flat-features. There is probably a good
    approach somewhere, that uses the ColumnTransformer, but the situation
    is made more complicated by the possibility of nested features in the
    Data/Container architecture.

    In any case, if I could have different transformers for different
    sets of features, I might avoid having to automatically infer which
    features or groups of features are categorical.
    """
    def process_and_wrap(train_dw, val_dw):
        train = train_dw.data
        val = val_dw.data

        x_train, x_val = process_container(
            train['x'], val['x'], processors)

        # x_train, x_val = process_container(
        #     train['x'], val['x'], processor, **processor_kwargs)
        new_train = Container(
            {
                'x': x_train,
                'y': train['y'],
                'index': train['index']
            },
            columns=train.columns,
            index=train.index,
            groups=train.groups,
            fits_in_memory=train.fits_in_memory
        )

        new_val = Container(
            {
                'x': x_val,
                'y': val['y'],
                'index': val['index']
            },
            columns=val.columns,
            index=val.index,
            groups=val.groups,
            fits_in_memory=val.fits_in_memory
        )
        # return new_train, new_val
        new_train_dw = copy(train_dw)
        new_val_dw = copy(val_dw)
        new_train_dw.data = new_train
        new_val_dw.data = new_val

        return new_train_dw, new_val_dw

    return process_and_wrap


def process_container(train, val, processors):
    train_dict = {}
    val_dict = {}

    for key, value in train.columns.items():
        if key in processors:
            if isinstance(value, dict):
                train_dict[key], val_dict[key] = process_container(
                    train[key], val[key], processors[key]
                )
            else:
                train_dict[key], val_dict[key] = process_data(
                    train[key], val[key], **processors[key]
                )
        else:
            train_dict[key], val_dict[key] = train[key], val[key]

    new_train = Container(
        train_dict,
        columns=train.columns,
        index=train.index,
        fits_in_memory=train.fits_in_memory
    )
    new_val = Container(
        val_dict,
        columns=val.columns,
        index=val.index,
        fits_in_memory=val.fits_in_memory
    )
    return new_train, new_val


def build_processor(processor, **kwargs):
    # TODO:
    # While I think this works, the column transformer will end up shuffling
    # the order of the columns - very annoying. Could be that the easiest
    # solution is to make my own column-transformer wrapper, which shuffles
    # the columns back again. Or something else. It's todo, anyway.
    if processor == 'Pipeline':
        p = Pipeline(
            steps=[
                (label, build_processor(transform, **transform_kwargs))
                for label, transform, transform_kwargs in kwargs['steps']
            ])
    elif processor == 'ColumnTransformer':
        if 'remainder' in kwargs:
            if isinstance(kwargs['remainder'], dict):
                remainder = build_processor(**kwargs['remainder'])
            else:
                remainder = kwargs['remainder']
        else:
            remainder = 'drop'

        p = ColumnTransformer(
            transformers=[
                (label, build_processor(transform, **transform_kwargs), cols)
                for label, transform, transform_kwargs, cols
                in kwargs['transformers']
            ],
            remainder=remainder
        )
    else:
        p = processor(**kwargs)

    return p


def process_data(train, val, processor, allow_categorical=False, **kwargs):
    train_data = train.as_numpy()
    cat, num = infer_categorical(train_data)
    if allow_categorical:
        cat = []
        num = list(range(train_data.shape[1]))

    p = build_processor(processor, **kwargs)
    p.fit(train_data[:, num])

    new_train_data = np.concatenate(
        [train_data[:, cat], p.transform(train_data[:, num])],
        axis=1
    )
    new_columns = np.array(train.columns)
    new_columns = list(new_columns[cat]) + list(new_columns[num])

    new_train = Data(
        new_train_data,
        columns=new_columns,
        index=train.index,
        dtype=train.dtype,
        fits_in_memory=train.fits_in_memory,
        groups=train.groups,
        predefined_splits=train.predefined_splits
    )

    val_data = val.as_numpy()
    new_val_data = np.concatenate(
        [val_data[:, cat], p.transform(val_data[:, num])],
        axis=1,
    )

    new_val = Data(
        new_val_data,
        columns=new_columns,
        index=val.index,
        dtype=val.dtype,
        fits_in_memory=val.fits_in_memory,
        groups=val.groups,
        predefined_splits=val.predefined_splits
    )
    return new_train, new_val


def pre_process_bootstrap_validation(split_number=0, random_state=42):
    def bootstrap(train, val):
        r = random.Random(random_state)
        n = len(val)
        return train, val.lazy_slice(r.choices(range(n), k=n))

    return bootstrap


class ECGData(Data):
    def __init__(self, data, mode='raw', index=None, dtype=tf.float32,
                 **kwargs):
        if mode not in {'raw', 'beat'}:
            mode = 'raw'

        self.mode = mode

        if index is None:
            with h5py.File(data, 'r') as f:
                index = np.array(range(len(f[mode])))

        super().__init__(
            data,
            index=index,
            dtype=dtype,
            columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'I', 'II'],
            **kwargs
        )

        if mode == 'beat':
            self._shape = [1200, 8]
        else:
            self._shape = [10000, 8]

    def __getitem__(self, item):
        with h5py.File(self.data, 'r') as f:
            return f[self.mode][self._index[item]]


class Extractor:
    def __init__(self, index=None, features=None, labels=None,
                 processing=None, fits_in_memory=None, cv_kwargs=None):
        self.index = {} if index is None else index
        self.features = {} if features is None else features
        self.labels = {} if labels is None else labels
        self.processing = {} if processing is None else processing
        self.fits_in_memory = fits_in_memory
        self.cv_kwargs = cv_kwargs

    def get_development_data(self) -> DataWrapper:
        raise NotImplementedError

    def get_test_data(self) -> DataWrapper:
        raise NotImplementedError


class Augmentor:
    def __init__(self, **settings):
        self.settings = settings

    def augment_training_data(self, data: DataWrapper) -> DataWrapper:
        raise NotImplementedError

    def augment_validation_data(self, data: DataWrapper) -> DataWrapper:
        return data


# Copy-pasted from pytorch 2.1, which isn't compatible yet with tensorflow
# https://pytorch.org/docs/2.1/_modules/torch/utils/data/dataset.html#StackDataset
class StackDataset(Dataset[T_stack]):
    r"""Dataset as a stacking of multiple datasets.

    This class is useful to assemble different parts of complex input data,
    given as datasets.

    Example:
        >>> # xdoctest: +SKIP
        >>> images = ImageDataset()
        >>> texts = TextDataset()
        >>> tuple_stack = StackDataset(images, texts)
        >>> tuple_stack[0] == (images[0], texts[0])
        >>> dict_stack = StackDataset(image=images, text=texts)
        >>> dict_stack[0] == {'image': images[0], 'text': texts[0]}

    Args:
        *args (Dataset): Datasets for stacking returned as tuple.
        **kwargs (Dataset): Datasets for stacking returned as dict.
    """
    datasets: Union[tuple, dict]

    def __init__(self, *args: Dataset[T_co], **kwargs: Dataset[T_co]) -> None:
        if args:
            if kwargs:
                raise ValueError(
                    "Supported either ``tuple``- (via ``args``) or"
                    "``dict``- (via ``kwargs``) like input/output, but"
                    " both types are given.")
            self._length = len(args[0])  # type: ignore[arg-type]
            if any(self._length != len(dataset) for dataset in args):
                raise ValueError("Size mismatch between datasets")
            self.datasets = args
        elif kwargs:
            tmp = list(kwargs.values())
            self._length = len(tmp[0])  # type: ignore[arg-type]
            if any(self._length != len(dataset) for dataset in tmp):
                raise ValueError("Size mismatch between datasets")
            self.datasets = kwargs
        else:
            raise ValueError("At least one dataset should be passed")

    def __getitem__(self, index):
        if isinstance(self.datasets, dict):
            return {k: dataset[index] for k, dataset in self.datasets.items()}
        return tuple(dataset[index] for dataset in self.datasets)

    def __len__(self):
        return self._length
