from enum import Enum

from sklearn.model_selection import KFold

from mim.fakes.fake_extractors import FakeExtractor
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
