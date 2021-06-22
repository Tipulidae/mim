from enum import Enum
from typing import NamedTuple, Any

import tensorflow as tf

import mim.experiments.hyper_parameter as hp
from mim.experiments.experiments import Experiment
from mim.experiments.search_strategies import Hyperband, RandomSearch
from mim.models.simple_nn import ecg_cnn
from mim.extractors.esc_trop import EscTrop
from mim.cross_validation import ChronologicalSplit
from mim.util.logs import get_logger
from mim.util.util import callable_to_string

log = get_logger("Hyper-experiments")


class HyperExperiment(NamedTuple):
    template: Experiment
    random_seed: int = 42
    strategy: Any = None
    strategy_kwargs: dict = None

    def run(self):
        log.info(
            f"Running hyper-experiment {self.name} using strategy "
            f"{self.strategy.__name__}"
        )
        searcher = self._searcher()
        searcher.search()

    def _searcher(self):
        return self.strategy(
            template=self.template,
            parent_base=self.__class__.__name__,
            parent_name=self.name,
            **self.strategy_kwargs
        )

    @property
    def experiments(self):
        return self._searcher().experiments()

    def asdict(self):
        return callable_to_string(self._asdict())

    @property
    def is_done(self):
        # TODO: Fix this!
        return False


class HyperSearch(HyperExperiment, Enum):
    R1_CNN_RS = HyperExperiment(
        template=Experiment(
            description="Testing ",
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0']
                },
                'index': {}
            },
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1/3
            },
            model=ecg_cnn,
            building_model_requires_development_data=True,
            optimizer={
                'name': tf.keras.optimizers.Adam,
                'kwargs': {'learning_rate': 1e-4}
            },
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            epochs=400,
            random_state=hp.Int(0, 1000000000),
            batch_size=128,
            model_kwargs={
                'cnn_kwargs': {
                    'dropout': hp.Choice([0.2, 0.3, 0.4, 0.5]),
                    'filter_first': hp.Int(16, 64),
                    'filter_last': hp.Int(16, 64),
                    'kernel_first': hp.Int(5, 31, step=2),
                    'kernel_last': hp.Int(5, 31, step=2),
                    'num_layers': hp.Choice([2, 3, 4]),
                    'dense': True,
                    'dense_size': 10,
                    'batch_norm': False,
                }
            },
        ),
        random_seed=42,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 200
        }
    )

    R1_CNN_HB = HyperExperiment(
        template=Experiment(
            description="",
            extractor=EscTrop,
            extractor_kwargs={
                "features": {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0']
                },
            },
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1/3
            },
            model=ecg_cnn,
            building_model_requires_development_data=True,
            optimizer={
                'name': tf.keras.optimizers.Adam,
                'kwargs': {'learning_rate': 1e-4}
            },
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            epochs=0,
            random_state=hp.Int(0, 1000000000),
            batch_size=128,
            model_kwargs={
                'cnn_kwargs': {
                    'dropout': hp.Choice([0.2, 0.3, 0.4, 0.5]),
                    'filter_first': hp.Int(16, 64),
                    'filter_last': hp.Int(16, 64),
                    'kernel_first': hp.Int(5, 31, step=2),
                    'kernel_last': hp.Int(5, 31, step=2),
                    'num_layers': hp.Choice([2, 3, 4]),
                    'dense': True,
                    'batch_norm': False
                }
            },
        ),
        random_seed=42,
        strategy=Hyperband,
        strategy_kwargs={
            'iterations': 100,
            'maximum_resource': 50,
            'resource_unit': 10
        }
    )

    R1_CNN_D100_HB = HyperExperiment(
        template=Experiment(
            description="",
            extractor=EscTrop,
            extractor_kwargs={
                "features": {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0']
                },
            },
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1/3
            },
            model=ecg_cnn,
            building_model_requires_development_data=True,
            optimizer={
                'name': tf.keras.optimizers.Adam,
                'kwargs': {'learning_rate': 1e-4}
            },
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            epochs=0,
            random_state=hp.Int(0, 1000000000),
            batch_size=128,
            model_kwargs={
                'cnn_kwargs': {
                    'dropout': hp.Choice([0.2, 0.3, 0.4, 0.5]),
                    'filter_first': hp.Int(16, 64),
                    'filter_last': hp.Int(16, 128),
                    'kernel_first': hp.Int(5, 31, step=2),
                    'kernel_last': hp.Int(5, 31, step=2),
                    'num_layers': hp.Choice([2, 3, 4]),
                    'downsample': True,
                    'dense': True,
                    'dense_size': 100,
                    'batch_norm': False
                },
                'dense_size': hp.Choice([10, 20, 30])
            },
        ),
        random_seed=42,
        strategy=Hyperband,
        strategy_kwargs={
            'iterations': 20,
            'maximum_resource': 50,
            'resource_unit': 10
        }
    )
