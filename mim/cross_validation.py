from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd

from mim.util.logs import get_logger

log = get_logger('Cross Validation')


class ClassBalance(metaclass=ABCMeta):
    def balance(self, x, indices) -> np.array:
        x = x.iloc[indices, :]
        positive = x[x['labels'] > 0].loc[:, 'index'].values
        negative = np.setdiff1d(indices, positive)

        balanced = self.sample(positive, negative)
        return balanced.sort()

    def sample(self, positive, negative) -> (np.array, np.array):
        return np.append(positive, negative)


class NullBalance(ClassBalance):
    def balance(self, x, indices):
        return indices


class DownSample(ClassBalance):
    def sample(self, positive, negative) -> (np.array, np.array):
        if len(negative) >= len(positive):
            negative = np.random.choice(negative, len(positive), replace=False)
        else:
            positive = np.random.choice(positive, len(negative), replace=False)

        return np.append(positive, negative)


class UpSample(ClassBalance):
    def sample(self, positive, negative):
        if len(negative) >= len(positive):
            positive = np.random.choice(positive, len(negative), replace=True)
        else:
            negative = np.random.choice(negative, len(positive), replace=True)

        return np.append(positive, negative)


class CrossValidator(metaclass=ABCMeta):
    @abstractmethod
    def split(self, x):
        pass


class KFold(CrossValidator):
    def __init__(self, n_splits, level='PatientID', random_state=700):
        """
        Perform k-fold cross-validation on a multi-indexed data set. The
        split can occur on one of the levels.

        For more general information about k-fold, see for example
        https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation

        :param n_splits: How many splits to use (the k in k-fold)
        :param level: The name of the multi-index level on which to split.
        Typically either PatientID or Time.
        """
        self.n_splits = n_splits
        self.level = level
        self.random_state = random_state

    def split(self, multi_index):
        """
        Given a multi-index, this returns an iterator that in each iteration
        gives a train and test split of the multi_index. The split itself
        is just a list of the row-numbers, which can be used with
        DataFrame.iloc.

        :param multi_index: The multi-index of the data set that should be
        split.
        :return: Iterator of all train, test splits.
        """
        index_df = pd.DataFrame(index=multi_index, columns=['index'])
        index_df['index'] = range(len(index_df))

        groups = self._make_groups(index_df)
        level_pos = multi_index.names.index(self.level)

        for i in range(len(groups)):
            group = sorted(groups[i, :])
            if level_pos == 0:
                test = index_df.loc[pd.IndexSlice[group, :], 'index'].values
            elif level_pos == 1:
                test = index_df.loc[pd.IndexSlice[:, group], 'index'].values
            else:
                test = group

            train = np.setdiff1d(index_df.loc[:, 'index'].values, test)
            yield train, test

    def _make_groups(self, index_df):
        random = np.random.RandomState(self.random_state)
        values = index_df.index.get_level_values(self.level).unique().values
        random.shuffle(values)
        padding = (self.n_splits - len(values) % self.n_splits) % self.n_splits

        values = np.append(values, np.zeros(padding, dtype=type(values[0])))
        groups = values.reshape(
            (self.n_splits, len(values) // self.n_splits),
            order='F')
        if padding > 0:
            groups = groups[:, :-1]

        groups.sort()
        return groups
