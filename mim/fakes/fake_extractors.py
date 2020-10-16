import pandas as pd
from sklearn.datasets import make_classification


class FakeExtractor:
    def __init__(self, specification=None):
        if specification['index']:
            self.kwargs = specification['index']
        else:
            self.kwargs = {}

    def get_data(self):
        X, y = make_classification(**self.kwargs)
        return pd.DataFrame(X), pd.DataFrame(y)
