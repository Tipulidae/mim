from enum import Enum
from typing import NamedTuple, Any

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

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
    validate_experiment_params: Any = None

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
            validator=self.validate_experiment_params,
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


def validate_pool_size(xp_kwargs, minimum_output_size=4):
    try:
        model_kwargs = xp_kwargs['model_kwargs']
    except KeyError:
        return True

    try:
        cnn_kwargs = model_kwargs['cnn_kwargs']
    except KeyError:
        return True

    try:
        mode = xp_kwargs['extractor_kwargs']['features']['ecg_mode']
    except KeyError:
        return True

    shape = 10000 if mode == 'raw' else 1200
    if 'downsample' in cnn_kwargs and cnn_kwargs['downsample']:
        shape //= 2

    num_layers = cnn_kwargs['num_layers']
    if 'pool_sizes' not in cnn_kwargs:
        pool_sizes = num_layers * [cnn_kwargs['pool_size']]
    else:
        pool_sizes = cnn_kwargs['pool_sizes']

    for pool in pool_sizes:
        shape //= pool

    if shape < minimum_output_size:
        return False

    if cnn_kwargs['kernel_last'] >= (shape * pool_sizes[-1]) / 2:
        return False

    return True


class HyperSearch(HyperExperiment, Enum):
    AMI_R1_CNN_RS = HyperExperiment(
        template=Experiment(
            description="Try to find good settings for predicting AMI using "
                        "a single raw ECG.",
            model=ecg_cnn,
            model_kwargs={
                'cnn_kwargs': hp.Choice([
                    {
                        'downsample': True,
                        'num_layers': num_layers,
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers),
                        'filter_first': hp.Int(8, 64, step=4),
                        'filter_last': hp.Int(8, 64, step=4),
                        'kernel_first': hp.Int(5, 65, step=4),
                        'kernel_last': hp.Int(5, 65, step=4),
                        'batch_norms': hp.Choices([True, False], k=num_layers),
                        'weight_decays': hp.Choices(
                            [1e-1, 1e-2, 1e-3, 0.0],
                            k=num_layers),
                        'ffnn_kwargs': hp.Choice([
                            None,
                            {
                                'sizes': hp.Choices([10, 50, 100], k=1),
                                'dropouts': hp.Choices(
                                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                                'batch_norms': [False]
                            }
                        ]),
                    } for num_layers in [2, 3, 4]
                ]),
                'ecg_ffnn_kwargs': None,
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.Choices([10, 20, 50, 100], k=1),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                        'batch_norms': [False]
                    }
                ]),
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0']
                },
                'labels': {'target': 'ami30'}
            },
            class_weight=hp.Choice([None, {0: 1, 1: 10.7}]),
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': {
                        'scheduler': PiecewiseConstantDecay,
                        'scheduler_kwargs': {
                            'boundaries': [153 * 50],
                            'values': hp.Choice([
                                [1e-2, 1e-3],
                                [1e-3, 1e-4],
                                [1e-4, 1e-5],
                                [1e-5, 1e-6]
                            ]),
                        }
                    },
                }
            },
            epochs=100,
            batch_size=64,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1/3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=42,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 500
        },
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
