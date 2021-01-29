import os
from enum import Enum

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mim.experiments.experiments import Experiment
from mim.model_wrapper import Model
from mim.config import PATH_TO_TEST_RESULTS


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

    def test_result_path_for_normal_experiment(self):
        expected_path = os.path.join(
            PATH_TO_TEST_RESULTS,
            'FakeExperiment',
            'default_experiment',
            'results.pickle'
        )

        assert FakeExperiment.default_experiment.result_path == expected_path

    def test_result_path_for_bare_experiment(self):
        bare = Experiment(
            description='This has no parent',
            alias='bare'
        )
        expected_path = os.path.join(
            PATH_TO_TEST_RESULTS,
            'Experiment',
            'bare',
            'results.pickle'
        )

        assert bare.result_path == expected_path

    def test_can_specify_parent_name_for_result_path(self):
        xp = Experiment(
            description='This has no parent, but I can pretend like it does',
            alias='foo',
            parent_base='HyperSearch',
            parent_name='RANDOM_SEARCH',
        )
        expected_path = os.path.join(
            PATH_TO_TEST_RESULTS,
            'HyperSearch',
            'RANDOM_SEARCH',
            'foo',
            'results.pickle'
        )

        assert xp.result_path == expected_path
