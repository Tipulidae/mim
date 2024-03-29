from enum import Enum

import numpy as np
from keras.optimizers import Adam

from mim.fakes.fake_extractors import FakeExtractor
from mim.models.simple_nn import basic_ff
from mim.util.metrics import sparse_categorical_accuracy
from mim.experiments.experiments import Experiment
from mim.experiments.results import ExperimentResult


class SmallTestExperiment(Experiment, Enum):
    test_fake_data = Experiment(
        description='Test of validate, with fake data',
        extractor=FakeExtractor,
        model_kwargs={'n_estimators': 10},
        save_model=False,
        save_results=False,
        log_environment=False  # True (which is default) is a little slow
    )

    test_keras = Experiment(
        description='Test small Keras network using fake data',
        extractor=FakeExtractor,
        extractor_index={
            'n_samples': 128,
            'n_features': 128,
            'n_informative': 100,
            'n_classes': 2,
            'random_state': 1111
        },
        model=basic_ff,
        model_kwargs={},
        batch_size=32,
        ignore_callbacks=True,
        epochs=2,
        optimizer=Adam,
        learning_rate=1e-2,
        loss='binary_crossentropy',
        metrics=['accuracy'],
        scoring=sparse_categorical_accuracy,
        log_environment=False,
        save_model=False,
        save_results=False
    )


class TestRunOneExperiment:
    def test_fake_experiment(self):
        result = SmallTestExperiment.test_fake_data.run(action='train')
        assert isinstance(result, ExperimentResult)
        assert result.num_splits_done == 5
        assert np.mean(result.validation_scores) > 0.8

    def test_keras(self):
        result = SmallTestExperiment.test_keras.run(action='train')
        assert np.mean(result.validation_scores) > 0
