from enum import Enum
from typing import NamedTuple, Any

import tensorflow as tf

import mim.experiments.hyper_parameter as hp
from mim.experiments.experiments import Experiment
from mim.experiments.search_strategies import Hyperband, RandomSearch
from mim.models.simple_nn import basic_cnn3, super_basic_cnn, basic_cnn
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
    SingleECG = HyperExperiment(
        template=Experiment(
            description="Experiment using variations of a basic CNN for "
                        "predicing MACE using a single ECG record.",
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'beat',
                    'ecgs': ['index']
                },
                'index': {},
            },

            model=basic_cnn3,
            building_model_requires_development_data=True,
            cv=ChronologicalSplit,
            cv_kwargs={'test_size': 1 / 3},
            optimizer={
                'name': hp.Choice([
                    tf.keras.optimizers.Adam,
                    tf.keras.optimizers.SGD]
                ),
                'kwargs': {
                    'learning_rate':
                        hp.Choice([3e-3, 1e-3, 3e-4, 1e-4])
                }
            },
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            epochs=0,
            batch_size=hp.Choice([32, 64, 128]),
            model_kwargs={
                'layers': hp.Choices(
                    data=[{
                        'filters': hp.Choice([8, 16, 32, 64]),
                        'kernel_size': hp.Choice([4, 8, 16, 32]),
                        'dropout': hp.Float(0.1, 0.8),
                    }],
                    k=hp.Int(2, 5)
                ),
                'pool_size': hp.Int(2, 7),
                'hidden_dropout': hp.Choice(
                    [None, hp.Float(0.1, 0.8)],
                )
            },

        ),
        random_seed=42,
        strategy=Hyperband,
        strategy_kwargs={
            'maximum_resource': 40,
            'resource_unit': 5
        }
    )

    LearningRates = HyperExperiment(
        template=Experiment(
            description="Experiment using very simple cnn, testing different "
                        "learning rates, batch sizes and dropout rates.",
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'beat',
                    'ecgs': ['index']
                },
                'index': {},
            },
            model=super_basic_cnn,
            building_model_requires_development_data=True,
            cv=ChronologicalSplit,
            cv_kwargs={'test_size': 1 / 3},
            optimizer={
                'name': tf.keras.optimizers.Adam,
                'kwargs': {
                    'learning_rate':
                        hp.Choice([3e-3, 1e-3, 3e-4, 1e-4, 3e-5])
                }
            },
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            epochs=200,
            random_state=hp.Int(0, 1000000000),
            batch_size=hp.Choice([8, 16, 32, 64, 128]),
            model_kwargs={
                'dropout': hp.Choice(
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                )
            },
        ),
        random_seed=42,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 1000
        }
    )

    ConvParams = HyperExperiment(
        template=Experiment(
            description="Experiment using very simple cnn, testing different "
                        "filters, kernel sizes and pool sizes.",
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'beat',
                    'ecgs': ['index']
                },
                'index': {},
            },
            model=super_basic_cnn,
            building_model_requires_development_data=True,
            cv=ChronologicalSplit,
            cv_kwargs={'test_size': 1 / 3},
            optimizer={
                'name': tf.keras.optimizers.Adam,
                'kwargs': {'learning_rate': 3e-4}
            },
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            epochs=400,
            random_state=hp.Int(0, 1000000000),
            batch_size=hp.Choice([128, 256]),
            model_kwargs={
                'dropout': 0.3,
                'filters': hp.Choice([8, 16, 32, 64, 128]),
                'kernel_size': hp.Choice([4, 8, 16, 32, 64]),
                'pool_size': hp.Choice([2, 3, 4, 5, 6, 7, 8, 9, 10])
            },
        ),
        random_seed=42,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 1000
        }
    )

    RawMultiLayer = HyperExperiment(
        template=Experiment(
            description="Testing ",
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['index']
                },
                'index': {}
            },
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1/3
            },
            model=basic_cnn,
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
                'dropout': 0.4,
                'filter_first': hp.Choice([16, 32, 64]),
                'filter_last': hp.Choice([16, 32, 64]),
                'kernel_first': hp.Choice([5, 15, 31]),
                'kernel_last': hp.Choice([5, 15, 31]),
                'num_layers': hp.Choice([2, 3, 4]),
                'dense': hp.Choice([True, False]),
                'batch_norm': hp.Choice([True, False])
            },
        ),
        random_seed=42,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 200
        }
    )

    RawMultiLayerHyperband = HyperExperiment(
        template=Experiment(
            description="",
            extractor=EscTrop,
            extractor_kwargs={
                "features": {
                    'ecg_mode': 'raw',
                    'ecgs': ['index']
                },
                "index": {},
            },
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1/3
            },
            model=basic_cnn,
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
                'dropout': hp.Choice([0.2, 0.3, 0.4, 0.5]),
                'filter_first': hp.Int(16, 64),
                'filter_last': hp.Int(16, 64),
                'kernel_first': hp.Int(5, 31, step=2),
                'kernel_last': hp.Int(5, 31, step=2),
                'num_layers': hp.Choice([2, 3, 4]),
                'dense': True,
                'batch_norm': False
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
