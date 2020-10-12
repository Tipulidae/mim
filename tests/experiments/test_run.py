from enum import Enum

from sklearn.model_selection import KFold

from mim.fakes.fake_extractors import FakeExtractor, MnistExtractor
from mim.fakes.fake_classifiers import Ann
from mim.metric_wrapper import sparse_categorical_accuracy
from mim.experiments.experiments import Experiment
from mim.experiments.run import run_one_experiment


class SmallTestExperiment(Experiment, Enum):
    test_fake_data = Experiment(
        description='Test of validate, with fake data',
        extractor=FakeExtractor,
        cv=KFold,
        cv_args={'n_splits': 2},
        params={
            'n_estimators': 10,
        },
    )

    test_keras_mnist = Experiment(
        description='Test Keras using mnist data',
        extractor=MnistExtractor,
        cv=KFold,
        cv_args={'n_splits': 2},
        algorithm=Ann,
        params={},
        scoring=sparse_categorical_accuracy,
    )


class TestRunOneExperiment:
    def test_fake_experiment(self):
        res = run_one_experiment(
            SmallTestExperiment.test_fake_data)

        assert 'predictions' in res
        assert 'train_score' in res
        assert 'test_score' in res
        assert 'feature_importance' in res
        assert 'fit_time' in res
        assert 'score_time' in res
        assert 'targets' in res
        assert 'history' in res

    def test_keras_mnist(self):
        res = run_one_experiment(
            SmallTestExperiment.test_keras_mnist
        )
        assert res['test_score'].mean() > 0.8
