from enum import Enum

from sklearn.multiclass import OneVsRestClassifier

from mim.experiments.experiments import Experiment
from mim.config import HyperParams
from mim.model_wrapper import RandomForestClassifier, LogisticRegression


class FakeExperiment(Experiment, Enum):
    default_experiment = Experiment('')

    has_description = Experiment(
        description='Test that the description works')

    no_default_algorithm = Experiment(
        description='Test that the default algorithm will be Random Forest')

    using_different_classifier = Experiment(
        description='Test that different classifiers can be used',
        algorithm=LogisticRegression,
        params={})

    has_custom_params_dict = Experiment(
        description='Test that different params can be specified',
        params={'n_estimators': 42})

    has_custom_params_enum = Experiment(
        description='Test that util.HyperParams enum can be used for params.',
        params=HyperParams.P1)

    uses_wrapper = Experiment(
        description='Test that a wrapper algorithm can be used to wrap the '
                    'basic classifier',
        wrapper=OneVsRestClassifier)


class TestExperiment:
    def test_description(self):
        xp = FakeExperiment.has_description
        assert 'Test that the description works' == xp.description

    def test_default_classifier_is_rf(self):
        xp = FakeExperiment.no_default_algorithm
        assert isinstance(xp.classifier, RandomForestClassifier)

    def test_different_classifier(self):
        xp = FakeExperiment.using_different_classifier
        assert isinstance(xp.classifier, LogisticRegression)

    def test_can_specify_different_params(self):
        xp = FakeExperiment.has_custom_params_dict
        assert 42 == xp.classifier.model.n_estimators

    def test_can_specify_params_as_enum(self):
        xp = FakeExperiment.has_custom_params_enum
        assert HyperParams.P1 == xp.params
        assert 1000 == xp.classifier.model.n_estimators
        assert 5 == xp.classifier.model.max_features

    def test_wrapper_wraps_classifier(self):
        xp = FakeExperiment.uses_wrapper
        assert isinstance(xp.classifier, RandomForestClassifier)
        assert isinstance(xp.classifier.model, OneVsRestClassifier)
