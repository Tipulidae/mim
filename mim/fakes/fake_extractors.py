from sklearn.datasets import make_classification
from tensorflow import float64

from mim.extractors.extractor import Container, Data, Extractor


class FakeExtractor(Extractor):
    def __init__(self, **kwargs):
        super().__init__(None, None, None, None)
        if 'index' in kwargs:
            if kwargs['index'] is not None:
                self.kwargs = kwargs['index']
            else:
                self.kwargs = {}
        else:
            self.kwargs = kwargs

    def get_data(self):
        x, y = make_classification(**self.kwargs)
        index = range(len(x))
        x = Data(x, index=index, dtype=float64)
        y = Data(y, index=index, dtype=float64)
        return Container.from_dict({'x': x, 'y': y})


class FakeECG(Extractor):
    def get_data(self):
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
        return Container({
            'x': Container(
                {'ecg': Data(x.reshape((n_samples, rows, cols)),
                             dtype=float64)}
            ),
            'y': Data(y)
        })
