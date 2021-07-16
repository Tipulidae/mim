from sklearn.datasets import make_classification
from tensorflow import float64

from mim.extractors.extractor import Container, Data, Extractor


class FakeExtractor(Extractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "index" in kwargs:
            self.mc_kwargs = kwargs["index"]
        else:
            self.mc_kwargs = {}

    def get_data(self) -> Container:
        x, y = make_classification(**self.mc_kwargs)
        index = range(len(x))
        x = Data(x, index=index, dtype=float64)
        y = Data(y, index=index, dtype=float64)
        return Container({'x': x, 'y': y, 'index': Data(list(index))})


class FakeECG(Extractor):
    def get_data(self,) -> Container:
        rows, cols = self.index['shape']
        n_features = rows * cols
        n_samples = self.index['n_samples']
        n_informative = int(self.index['informative_proportion'] * n_features)
        x, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_classes=self.index['n_classes'],
            random_state=1234
        )

        c = Container({
            'x': Container(
                {'ecg': Data(x.reshape((n_samples, rows, cols)),
                             dtype=float64)}
            ),
            'y': Data(y)
        })

        return c
