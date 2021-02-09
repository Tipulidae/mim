from enum import Enum

from mim.fakes.fake_extractors import FakeExtractor
from mim.models.simple_nn import basic_ff
from mim.metric_wrapper import sparse_categorical_accuracy
from mim.experiments.experiments import Experiment
from mim.experiments.run import run_one_experiment


class SmallTestExperiment(Experiment, Enum):
    test_fake_data = Experiment(
        description='Test of validate, with fake data',
        extractor=FakeExtractor,
        model_kwargs={'n_estimators': 10},
    )

    test_keras = Experiment(
        description='Test small Keras network using fake data',
        extractor=FakeExtractor,
        extractor_kwargs={
            "index": {
                'n_samples': 512,
                'n_features': 128,
                'n_informative': 100,
                'n_classes': 10,
                'random_state': 1111
            },
        },
        model=basic_ff,
        model_kwargs={},
        batch_size=32,
        ignore_callbacks=True,
        epochs=2,
        optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        scoring=sparse_categorical_accuracy,
    )


class TestRunOneExperiment:
    def test_fake_experiment(self):
        res = run_one_experiment(SmallTestExperiment.test_fake_data)

        assert 'predictions' in res
        assert 'train_score' in res
        assert 'test_score' in res
        assert 'feature_importance' in res
        assert 'fit_time' in res
        assert 'score_time' in res
        assert 'targets' in res
        assert 'history' in res

    def test_keras(self):
        res = run_one_experiment(SmallTestExperiment.test_keras)
        assert res['test_score'].mean() > 0
