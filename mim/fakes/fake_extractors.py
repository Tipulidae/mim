import numpy as np
from sklearn.datasets import make_classification

from mim.experiments.extractor import DataWrapper, Extractor, Data


class FakeExtractor(Extractor):
    def get_test_data(self) -> DataWrapper:
        raise NotImplementedError

    def get_development_data(self) -> DataWrapper:
        x, y = make_classification(**self.index)
        n_samples, n_features = x.shape
        feature_names = [f"x{i}" for i in range(n_features)]
        index = list(range(n_samples))
        data = DataWrapper(
            features=Data(x.astype(np.float32), columns=feature_names),
            labels=Data(y.astype(np.float32), columns=['y']),
            index=Data(index, columns=['index']),
            fits_in_memory=self.fits_in_memory
        )
        return data
