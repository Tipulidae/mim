from mim.extractors.extractor import Data
from mim.fakes.fake_extractors import FakeExtractor


def test_data_is_correct_format():
    data = FakeExtractor().get_data()
    assert isinstance(data, Data)
    assert data['x'].as_numpy.shape == (100, 20)
    assert data['y'].as_numpy.shape == (100,)


def test_can_specify_inputs():
    ext = FakeExtractor(
        index=dict(
            n_samples=10,
            n_features=5,
            n_informative=3
        )
    )
    data = ext.get_data()
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
    data = FakeExtractor(**extractors_kwargs).get_data()
    assert data['x'].as_numpy.shape == (10, 5)
