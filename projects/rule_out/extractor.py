from math import pi
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from mim.experiments.extractor import Extractor, DataWrapper, Data
from mim.util.logs import get_logger


log = get_logger("Rule-out extractor")


def make_cone(n_samples=1000, center=(0.0, 0.0), scale=(1.0, 1.0),
              random_generator=None):
    if random_generator is None:
        random_generator = np.random.default_rng()
    # r = np.random.default_rng(random_state)
    rs = random_generator.uniform(low=0.0, high=1.0, size=(n_samples,))
    args = random_generator.uniform(low=0.0, high=2*pi, size=(n_samples,))
    data = np.stack([rs * np.cos(args), rs * np.sin(args)], axis=1)
    data = data * np.array(scale) + np.array(center)
    return data


def make_double_cones(
        negatives_count=1000,
        positives_count=1000,
        negatives_center=(0.0, 0.0),
        positives_center=(1.0, 0.0),
        negatives_scale=(1.0, 1.0),
        positives_scale=(1.0, 1.0),
        shuffle=False,
        random_state=41,
        **_):
    r = np.random.default_rng(random_state)
    x1 = make_cone(
        n_samples=negatives_count,
        center=negatives_center,
        scale=negatives_scale,
        random_generator=r
    )
    x2 = make_cone(
        n_samples=positives_count,
        center=positives_center,
        scale=positives_scale,
        random_generator=r
    )
    y1 = np.zeros(shape=(negatives_count,))
    y2 = np.ones(shape=(positives_count,))
    x = np.concatenate([x1, x2])
    # x = np.stack([x1, x2], axis=1)
    y = np.concatenate([y1, y2])

    index = range(positives_count + negatives_count)
    if shuffle:
        index = r.permutation(index)

    return x[index, :], y[index]


class Cones(Extractor):
    def get_development_data(self) -> DataWrapper:
        if 'random_state' in self.index:
            random_state = self.index['random_state']
        else:
            random_state = 97

        x, y = make_double_cones(
            random_state=random_state,
            shuffle=False,
            **self.features,
        )
        cols = ['x1', 'x2']
        if 'cheat' in self.features and self.features['cheat']:
            d1 = (x[:, 0] ** 2 + (x[:, 1] - 1) ** 2) ** 0.5
            d2 = (x[:, 0] ** 2 + (x[:, 1] + 1) ** 2) ** 0.5
            c1 = np.where(d1 > 2, 0, 1)
            c2 = np.where(d2 > 2, 0, 1)
            x = np.hstack((
                x, d1.reshape(-1, 1),
                d2.reshape(-1, 1), c1.reshape(-1, 1), c2.reshape(-1, 1)
            ))
            cols.extend([
                'd1', 'd2',
                'c1', 'c2'
            ])

        data = DataWrapper(
            features=Data(x, columns=cols),
            labels=Data(y, columns=['y']),
            index=Data(range(len(y)), columns=['index']),
            fits_in_memory=True,
        )
        return data

    def get_test_data(self) -> DataWrapper:
        pass


class Blobs(Extractor):
    def get_development_data(self) -> DataWrapper:
        n_samples = (self.features['negatives_counts'] +
                     self.features['positives_counts'])
        centers = (self.features['negatives_centers'] +
                   self.features['positives_centers'])
        stds = self.features['negatives_std'] + self.features['positives_std']
        if 'random_state' in self.index:
            random_state = self.index['random_state']
        else:
            random_state = 97

        x, y = make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=stds,
            random_state=random_state,
        )
        negative_blobs = len(self.features['negatives_centers'])
        y_map = defaultdict(
            lambda: 1, {k: 0 for k in range(negative_blobs)})
        y = pd.Series(y).map(y_map).values

        num_features = x.shape[1]
        # if 'random_state' in self.index:
        #     random_state = self.index['random_state']
        # else:
        #     random_state = 97
        #
        # n_samples = self.index['n_samples']
        # if 'weights' in self.labels:
        #     weights = self.labels['weights']
        # else:
        #     weights = (0.5, 0.5)
        #
        # n_samples = [int(n_samples * weights[0]),
        # int(n_samples * weights[1])]
        #
        # x, y = make_blobs(
        #     n_samples=n_samples,
        #     n_features=self.features['n_features'],
        #     centers=self.features['centers'],
        #     cluster_std=self.features['cluster_std'],
        #     random_state=random_state,
        # )
        #
        data = DataWrapper(
            features=Data(x, columns=[
                f'x{i}' for i in range(num_features)]),
            labels=Data(y, columns=['y']),
            index=Data(range(len(y)), columns=['index']),
            fits_in_memory=True,
        )

        return data

    def get_test_data(self) -> DataWrapper:
        pass
