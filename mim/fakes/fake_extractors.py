from sklearn.datasets import make_classification
from tensorflow import float64

from mim.extractors.extractor import Container, Data, Extractor, \
    DataProvider, SingleContainerLinearSplitProvider


class FakeExtractor(Extractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "index" in kwargs:
            self.mc_kwargs = kwargs["index"]
        else:
            self.mc_kwargs = {}
        # self.kwargs = kwargs
        # if 'specification' in kwargs:
        # self.kwargs = {}
        # if 'index' in kwargs:
        #     if kwargs['index'] is not None:
        #         self.kwargs = kwargs['index']
        #     else:
        #         self.kwargs = {}
        # else:
        #     self.kwargs = kwargs

    def get_data_provider(self, dp_kwargs) -> DataProvider:
        x, y = make_classification(**self.mc_kwargs)
        index = range(len(x))
        x = Data(x, index=index, dtype=float64)
        y = Data(y, index=index, dtype=float64)
        # c = Container.from_dict({'x': x, 'y': y})
        c = Container({'x': x, 'y': y})
        return SingleContainerLinearSplitProvider(c, **dp_kwargs)
