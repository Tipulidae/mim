import random
from copy import copy

import numpy as np
import tensorflow as tf
import h5py

from typing import Dict


class Data:
    def __init__(
            self,
            data,
            index=None,
            columns=None,
            dtype=tf.int64,
            fits_in_memory=True,
            groups=None,
            predefined_splits=None):
        self.data = data
        if columns is None:
            self._columns = [0]
        else:
            self._columns = columns
        self.dtype = dtype
        self.groups = groups
        self.predefined_splits = predefined_splits
        self._fits_in_memory = fits_in_memory
        if index is None:
            self._index = np.array(range(len(data)))
        else:
            self._index = index

        self._shape = infer_shape(data)

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


def infer_shape(data):
    shape = np.shape(data)
    n = len(shape)
    if n == 1:
        return []
    elif n < 1:
        return None

    return list(shape[1:])


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
                 processing=None, fits_in_memory=True, cv_kwargs=None):
        self.index = {} if index is None else index
        self.features = {} if features is None else features
        self.labels = {} if labels is None else labels
        self.processing = {} if processing is None else processing
        self.fits_in_memory = fits_in_memory
        self.cv_kwargs = cv_kwargs

    def get_data(self) -> Container:
        raise NotImplementedError
