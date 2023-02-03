from sklearn.datasets import make_classification

from mim.experiments.extractor import DataWrapper, Extractor, Data


class FakeExtractor(Extractor):
    def get_test_data(self) -> DataWrapper:
        raise NotImplementedError

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "index" in kwargs:
            self.mc_kwargs = kwargs["index"]
        else:
            self.mc_kwargs = {}

    def get_development_data(self) -> DataWrapper:
        x, y = make_classification(**self.mc_kwargs)
        n_samples, n_features = x.shape
        feature_names = [f"x{i}" for i in range(n_features)]
        index = list(range(n_samples))
        data = DataWrapper(
            features=Data(x, columns=feature_names),
            labels=Data(y, columns=['y']),
            index=Data(index, columns=['index'])
        )
        return data
