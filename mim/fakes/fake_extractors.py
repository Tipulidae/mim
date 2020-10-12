import pandas as pd
from sklearn.datasets import make_classification
from tensorflow import keras


class FakeExtractor:
    def __init__(self, *args, **kwargs):
        pass

    def get_data(self):
        X, y = make_classification()
        return pd.DataFrame(X), pd.DataFrame(y)


class MnistExtractor:
    def __init__(self, *args, **kwargs):
        pass

    def get_data(self):
        (X, y), _ = keras.datasets.mnist.load_data()
        X = X.reshape(60000, 784).astype("float32") / 255
        y = y.astype("float32")
        return pd.DataFrame(X[:6000]), pd.DataFrame(y[:6000])
