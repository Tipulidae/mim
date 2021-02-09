from copy import copy
from sklearn.model_selection import KFold

import numpy as np
import tensorflow as tf
import h5py

from typing import Dict, Tuple, Iterator


class Data:
    def __init__(self, data, index=None, dtype=tf.int64, groups=None):
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
    def __init__(self, data: Dict[str, Data], index=None, **kwargs):
        if not isinstance(data, dict):
            raise TypeError(f"Data must be of type dict (was {type(data)})")
        if len({len(v) for v in data.values()}) != 1:
            raise ValueError("Inconsistent length of constituent Data")
        if index is None:
            index = next(iter(data.values())).index
        super().__init__(data, index=index, **kwargs)

    def lazy_slice(self, index):
        return self.__class__(
            {key: value.lazy_slice(index) for key, value in self.data.items()},
            dtype=self.type
        )

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


class Extractor:
    def __init__(self, index=None, features=None, labels=None,
                 processing=None, cv_kwargs=None):
        self.index = index
        self.features = features
        self.labels = labels
        self.processing = processing
        self.cv_kwargs = cv_kwargs

    def get_data_provider(self, dp_kwargs) -> "DataProvider":
        raise NotImplementedError


class DataProvider:
    def __init__(self, mode, cv_folds, cv_set):
        self.mode = mode
        self.cv_folds = cv_folds
        self.cv_set = cv_set

    def get_set(self, name) -> Container:
        raise NotImplementedError

    def _get_cv(self, data) -> Iterator[Tuple[Container, Container]]:
        x = data.index
        y = data['y'].as_numpy
        groups = data.groups
        cv = KFold(n_splits=self.cv_folds)
        for train, val in cv.split(x, y=y, groups=groups):
            yield data.split(train, val)

    def split(self) -> Iterator[Tuple[Container, Container]]:
        if self.mode == "cv":
            return self._get_cv(self.get_set(self.cv_set))
        elif self.mode == "train_val":
            return self.train_val_split()

    def train_val_split(self) -> Iterator[Tuple[Container, Container]]:
        yield self.get_set("train"), self.get_set("val")


class SingleContainerLinearSplitProvider(DataProvider):
    def __init__(self, container: Container, train_frac: float,
                 val_frac: float, test_frac: float, **kwargs):
        super().__init__(mode=kwargs["mode"], cv_folds=kwargs["cv_folds"],
                         cv_set=kwargs["cv_set"])
        self.container = container
        assert train_frac + val_frac + test_frac == 1.0
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.n = len(self.container)
        self.train_val_point = int(self.n * self.train_frac)
        self.val_test_point = self.train_val_point + int(self.n*self.val_frac)

    def get_set(self, name) -> Container:
        if name == "all":
            return self.container
        elif name == "train":
            return self.container.lazy_slice(range(self.train_val_point))
        elif name == "val":
            return self.container.lazy_slice(range(self.train_val_point,
                                                   self.val_test_point))
        elif name == "test":
            return self.container.lazy_slice(range(self.val_test_point,
                                                   self.n))
        elif name == "dev":
            return self.container.lazy_slice(range(self.val_test_point))
        else:
            print("Throw some error here")


class IndividualContainerDataProvider(DataProvider):
    def __init__(self, container_dict: Dict[str, Container], **kwargs):
        super().__init__(mode=kwargs["mode"], cv_folds=kwargs["cv_folds"],
                         cv_set=kwargs["cv_set"])
        self.container_dict = container_dict

    def get_set(self, name) -> Container:
        return self.container_dict[name]
