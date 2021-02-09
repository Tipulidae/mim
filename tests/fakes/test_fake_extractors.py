from mim.extractors.extractor import Data
from mim.fakes.fake_extractors import FakeExtractor

DEFAULT_SPLIT_KWARGS = {"train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2,
                        "mode": "cv", "cv_folds": 5, "cv_set": "all"}


def test_data_is_correct_format():
    dp = FakeExtractor().get_data_provider(DEFAULT_SPLIT_KWARGS)
    data = dp.get_set("all")
    assert isinstance(data, Data)
    assert data['x'].as_numpy.shape == (100, 20)
    assert data['y'].as_numpy.shape == (100,)


def test_can_specify_inputs():
    ext = FakeExtractor(**{"index": dict(n_samples=10,
                                         n_features=5,
                                         n_informative=3)})
    dp = ext.get_data_provider(DEFAULT_SPLIT_KWARGS)
    data = dp.get_set("all")

    assert data['x'].as_numpy.shape == (10, 5)


def test_can_specify_inputs_like_experiment():
    index = {'n_samples': 10, 'n_features': 5}
    features = {'foo': 43, 'bar': 999}
    labels = [123]
    processing = {1, 2, 3}
    extractors_kwargs = dict(
        index=index,
        features=features,
        labels=labels,
        processing=processing
    )
    dp = FakeExtractor(**extractors_kwargs).get_data_provider(
        DEFAULT_SPLIT_KWARGS)

    data = dp.get_set("all")

    assert data['x'].as_numpy.shape == (10, 5)
