from enum import Enum

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mim.experiments.experiments import Experiment
from mim.model_wrapper import Model


class FakeExperiment(Experiment, Enum):
    default_experiment = Experiment('')

    has_description = Experiment(
        description='Test that the description works'
    )

    no_default_algorithm = Experiment(
        description='Test that the default algorithm will be Random Forest'
    )

    using_different_classifier = Experiment(
        description='Test that different classifiers can be used',
        model=LogisticRegression,
        model_kwargs={}
    )

    has_custom_params_dict = Experiment(
        description='Test that different params can be specified',
        model_kwargs={'n_estimators': 42}
    )


class TestExperiment:
    def test_description(self):
        xp = FakeExperiment.has_description
        assert 'Test that the description works' == xp.description

    def test_default_classifier_is_rf(self):
        xp = FakeExperiment.no_default_algorithm
        model = xp.get_model(None, None)
        assert isinstance(model, Model)
        assert isinstance(model.model, RandomForestClassifier)

    def test_different_classifier(self):
        xp = FakeExperiment.using_different_classifier
        model = xp.get_model(None, None)
        assert isinstance(model, Model)
        assert isinstance(model.model, LogisticRegression)

    def test_can_specify_different_params(self):
        xp = FakeExperiment.has_custom_params_dict
        assert 42 == xp.get_model(None, None).model.n_estimators
