from sklearn.datasets import make_classification
from tensorflow import float64

from mim.extractors.extractor import Container, Data, Extractor


class FakeExtractor(Extractor):
    def __init__(self, **kwargs):
        super().__init__(None, None, None, None)
        # self.kwargs = kwargs
        # if 'specification' in kwargs:
        # self.kwargs = {}
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
