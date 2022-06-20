from math import ceil

from sklearn.model_selection import PredefinedSplit

from mim.extractors.extractor import Container
from mim.util.logs import get_logger

log = get_logger('Cross Validation')


class CrossValidationWrapper:
    def __init__(self, cv):
        self.cv = cv

    def split(self, data: Container):
        x = data.index
        y = data['y'].as_numpy()

        groups = data.groups
        for train, val in self.cv.split(x, y=y, groups=groups):
            yield data.split(train, val)


class ChronologicalSplit:
    def __init__(self, test_size=0.5):
        self.test_size = test_size

    def split(self, x, y=None, groups=None):
        n = len(x)
        k = ceil((1 - self.test_size) * n)
        train = list(range(k))
        test = list(range(k, n))
        yield train, test


class PredefinedSplitsRepeated:
    def __init__(self, predefined_splits, repeats=5):
        self.predefined_splits = predefined_splits
        self.repeats = repeats

    def split(self, x, y=None, groups=None):
        for i in range(self.repeats):
            ps = PredefinedSplit(self.predefined_splits)
            for train, test in ps.split(x, y, groups):
                yield train, test
