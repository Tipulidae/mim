import mim.model_wrapper as mw
from mim.fakes.fake_extractors import FakeExtractor


class TestModel:
    def test_rf_binary_classification_output_is_one_dimensional(self):
        data = FakeExtractor().get_data()
        # data = dp.get_set("all")
        clf = mw.RandomForestClassifier(n_estimators=10)
        clf.fit(data)
        assert clf.only_last_prediction_column_is_used

        prediction = clf.predict(data['x'])['prediction']
        assert 1 == len(prediction.columns)

    def test_rf_multiclass_classification_output_is_multi_dimensional(self):
        data = FakeExtractor(
            **{"index": dict(n_informative=3, n_classes=3)}
        ).get_data()
        # data = dp.get_set("all")
        clf = mw.RandomForestClassifier(n_estimators=10)
        clf.fit(data)
        assert not clf.only_last_prediction_column_is_used

        prediction = clf.predict(data['x'])['prediction']
        assert 3 == len(prediction.columns)

    def test_models_have_default_random_states(self):
        rf = mw.RandomForestClassifier()
        assert rf.model.random_state is not None

        rf = mw.RandomForestClassifier(random_state=8382)
        assert 8382 == rf.model.random_state

        rf = mw.RandomForestClassifier(random_state=3141592)
        assert 3141592 == rf.model.random_state
