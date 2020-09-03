import pandas as pd
from sklearn.datasets import make_classification


def make_fake_data(start='2015-01-01', end='2015-03-01'):
    index = make_fake_multi_index(list(range(10)), start=start, end=end)
    X, y = make_classification(n_samples=len(index), n_features=10,
                               n_redundant=8, class_sep=0.02, flip_y=0.01,
                               random_state=99)

    X = pd.DataFrame(X, index=index)
    y = pd.DataFrame(y, index=index)

    return X, y


def make_fake_data_drop_last_labels():
    X, y = make_fake_data()
    y = y.loc[pd.IndexSlice[:'2015-02-01', :], :]
    return X, y


def make_fake_multi_index(pids, start='2015-01-01', end='2015-03-01'):
    dates = pd.date_range(start=start, end=end, freq='1W-SUN')
    index = pd.MultiIndex.from_product(
        (dates, pids), names=['Time', 'PatientID'])

    return index
