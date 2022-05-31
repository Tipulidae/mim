from sklearn.model_selection import PredefinedSplit

from mim.extractors.extractor import DataWrapper
from mim.util.util import infer_categorical
from mim.util.logs import get_logger

log = get_logger('Cross Validation')


class CrossValidationWrapper:
    def __init__(self, cv):
        self.cv = cv

    def split(self, data: DataWrapper):
        x = data.index
        y = targets_for_stratification(data.y.values)

        groups = data.groups
        for train, val in self.cv.split(x, y=y, groups=groups):
            yield data.split(train, val)


def targets_for_stratification(data):
    """
    Given a data array, get only those columns that
    are reasonable to use for stratifying on in a cross-validation split.
    Specifically, if the target is multi-dimensional and some of the
    dimensions are non-categorical, we don't want to use them for
    stratification (sklearn will raise an error).

    Here, I first get the full target matrix, and if it's just a single
    vector, turn it to a column-vector. Then I check which columns are
    categorical, and return the slice that is. If no columns are categorical,
    return None. It will be an error to use a stratified splitter on such
    data.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)  # Turn to column vector

    cat, _ = infer_categorical(data)
    if len(cat) == 0:
        return None
    else:
        return data[:, cat]


class ChronologicalSplit:
    def __init__(self, test_size=0.5):
        self.test_size = test_size

    def split(self, x, y=None, groups=None):
        n = len(x)
        k = int((1 - self.test_size) * n)
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
