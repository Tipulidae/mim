from sklearn.ensemble import RandomForestClassifier

from mim.model_wrapper import Model
from mim.fakes.fake_extractors import FakeExtractor


class TestModel:
    def test_rf_binary_classification_output_is_one_dimensional(self):
        data = FakeExtractor().get_development_data()
        clf = Model(RandomForestClassifier(n_estimators=10))
        clf.fit(data)
        assert clf.only_last_prediction_column_is_used

        prediction = clf.predict(data)
        assert len(prediction.columns) == 1

    # TODO: this doesn't work yet because the DataWrapper doesn't know
    # how to properly deal with multiclass classification labels, so
    # when the prediction is turned into a DataFrame, the number of columns
    # are wrong. I should probably make some way to specify directly the
    # type of label when creating the DataWrapper, which would then make it
    # easy to solve this issue. But because I don't need multiclass
    # classification at the moment, I postpone that feature for now.
    # def test_rf_multiclass_classification_output_is_multi_dimensional(self):
    #     data = FakeExtractor(
    #         **{"index": dict(n_informative=3, n_classes=3)}
    #     ).get_data()
    #     # data = dp.get_set("all")
    #     clf = Model(RandomForestClassifier(n_estimators=10))
    #     clf.fit(data)
    #     assert not clf.only_last_prediction_column_is_used
    #
    #     print(f"{data.as_numpy()=}")
    #     prediction = clf.predict(data)
    #     assert len(prediction.columns) == 3

    def test_models_have_default_random_states(self):
        rf = Model(RandomForestClassifier())
        assert rf.model.random_state is not None

        rf = Model(RandomForestClassifier(), random_state=8382)
        assert rf.model.random_state == 8382

        rf = Model(RandomForestClassifier(), random_state=3141592)
        assert rf.model.random_state == 3141592
