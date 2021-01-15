from copy import copy

import numpy as np
import tensorflow as tf
import h5py


class Data:
    def __init__(self, data, index=None, dtype=tf.int64, groups=None,
                 **kwargs):
        self.data = data
        self.dtype = dtype
        self.groups = groups
        if index is None:
            self._index = range(len(data))
        else:
            self._index = index

        self._shape = infer_shape(data)

    def split(self, index_a, index_b):
        return self.lazy_slice(index_a), self.lazy_slice(index_b)

    def lazy_slice(self, index):
        new_data = copy(self)
        new_data._index = [self._index[i] for i in index]
        return new_data

    @property
    def index(self):
        return range(len(self))

    @property
    def as_dataset(self):
        return tf.data.Dataset.from_generator(
            self,
            output_types=self.type,
            output_shapes=self.shape
        )

    @property
    def as_numpy(self):
        return np.array(list(self))

    @property
    def type(self):
        return self.dtype

    @property
    def shape(self):
        return tf.TensorShape(self._shape)

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def lazy_slice(self, index):
        return self.__class__(
            {key: value.lazy_slice(index) for key, value in self.data.items()},
            index=index,
            dtype=self.type
        )

    @classmethod
    def from_dict(cls, data_dict, index):
        def to_data(item):
            if isinstance(item, Data):
                return item
            else:
                return Data(item, index=index)

        return cls(
            {key: to_data(value) for key, value in data_dict.items()},
            index=index
        )

    @property
    def type(self):
        return {key: value.type for key, value in self.data.items()}

    @property
    def shape(self):
        return {key: value.shape for key, value in self.data.items()}

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
                index = range(len(f[mode]))

        super().__init__(data, index=index, dtype=dtype, **kwargs)

        if mode == 'beat':
            self._shape = [1200, 8]
        else:
            self._shape = [10000, 8]

    def __getitem__(self, item):
        with h5py.File(self.data, 'r') as f:
            return f[self.mode][self._index[item]]
