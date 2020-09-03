import pandas as pd
from sklearn.datasets import make_classification


class FakeExtractor:
    def __init__(self, *args, **kwargs):
        pass

    def get_data(self):
        X, y = make_classification()
        return pd.DataFrame(X), pd.DataFrame(y)
