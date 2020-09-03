import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import QuantileTransformer

import mim.model_wrapper as mw
from mim.fakes.fake_classifiers import RandomRegressor


class TestModel:
    def test_input_to_fit_is_dataframe(self):
        X, y = make_classification()
        X = pd.DataFrame(X)
        y = pd.DataFrame(y, index=X.index)

        clf = mw.RandomForestClassifier(n_estimators=10)
        clf.fit(X, y)

    def test_rf_returns_correct_number_of_classes(self):
        X2, y2 = make_classification()
        X2 = pd.DataFrame(X2)
        y2 = pd.DataFrame(y2, index=X2.index)

        X3, y3 = make_classification(n_informative=3, n_classes=3)
        X3 = pd.DataFrame(X3)
        y3 = pd.DataFrame(y3, index=X3.index)

        clf = mw.RandomForestClassifier(n_estimators=10)
        clf.fit(X2, y2)
        assert clf.only_last_prediction_column_is_used
        assert 1 == len(clf.predict(X2)['prediction'].columns)

        clf.fit(X3, y3)
        assert not clf.only_last_prediction_column_is_used
        assert 3 == len(clf.predict(X3)['prediction'].columns)

    def test_prediction_is_dataframe_in_dict(self):
        X, y = make_classification()
        X = pd.DataFrame(X)
        y = pd.DataFrame(y, index=X.index)

        clf = mw.RandomForestClassifier(n_estimators=10)

        clf.fit(X, y)
        prediction = clf.predict(X)

        assert isinstance(prediction, dict)
        assert 'prediction' in prediction
        assert isinstance(prediction['prediction'], pd.DataFrame)
        assert 1 == len(prediction)

    def test_regressor_uses_all_prediction_columns(self):
        rf = mw.RandomForestRegressor()
        assert not rf.only_last_prediction_column_is_used

        rr = RandomRegressor()
        assert not rr.only_last_prediction_column_is_used

    def test_regression_model_can_use_transformer(self):
        X, y = make_regression()
        X = pd.DataFrame(X)
        y = pd.DataFrame(y, index=X.index)

        rf = mw.RandomForestRegressor(
            transformer=QuantileTransformer,
            transformer_args={
                'output_distribution': 'normal',
                'n_quantiles': 50
            },
            n_estimators=10)

        rf.fit(X, y)
        p = rf.predict(X)

        assert 'prediction' in p

    def test_models_have_default_random_states(self):
        rf = mw.RandomForestClassifier()
        assert rf.model.random_state is not None

        rf = mw.RandomForestClassifier(random_state=8382)
        assert 8382 == rf.model.random_state

        rf = mw.RandomForestClassifier(random_state=3141592)
        assert 3141592 == rf.model.random_state
