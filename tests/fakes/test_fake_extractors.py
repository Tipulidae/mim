from mim.extractors.extractor import DataWrapper
from mim.fakes.fake_extractors import FakeExtractor


class TestFakeExtractor:
    def test_correct_return_type(self):
        data = FakeExtractor().get_development_data()
        assert isinstance(data, DataWrapper)

    def test_arrays_get_right_shape(self):
        data = FakeExtractor().get_development_data()
        x, y = data.as_numpy()
        assert x.shape == (100, 20)
        assert y.shape == (100,)

    def test_can_specify_inputs(self):
        ext = FakeExtractor(
            index=dict(
                n_samples=10,
                n_features=5,
                n_informative=3
            )
        )
        data = ext.get_development_data()
        x, y = data.as_numpy()
        assert x.shape == (10, 5)
        assert y.shape == (10,)

    def test_can_specify_inputs_like_experiment(self):
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
        data = FakeExtractor(**extractors_kwargs).get_development_data()
        x, y = data.as_numpy()
        assert x.shape == (10, 5)
        assert y.shape == (10,)
