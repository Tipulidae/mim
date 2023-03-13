from collections import defaultdict

import pandas as pd
from sklearn.datasets import make_blobs

from mim.experiments.extractor import Extractor, DataWrapper, Data
from mim.util.logs import get_logger


log = get_logger("Rule-out extractor")


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
