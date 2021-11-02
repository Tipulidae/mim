from enum import Enum
from typing import NamedTuple, Any

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from sklearn.preprocessing import StandardScaler

import mim.experiments.hyper_parameter as hp
from mim.experiments.experiments import Experiment
from mim.experiments.search_strategies import Hyperband, RandomSearch
from mim.models.simple_nn import ecg_cnn, pretrained_resnet, ffnn
from mim.extractors.esc_trop import EscTrop
from mim.extractors.extractor import sklearn_process
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
            random_seed=self.random_seed,
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
    AMI_R1_CNN_HB = HyperExperiment(
        template=Experiment(
            description="Try to find good settings for predicting AMI using "
                        "a single raw ECG. Hyperband searcher.",
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
            class_weight={0: 1, 1: 10.7},
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
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
        strategy=Hyperband,
        strategy_kwargs={
            'iterations': 20,
            'maximum_resource': 40,
            'resource_unit': 5
        }
    )

    M_R1_FF_CNN_RS = HyperExperiment(
        template=Experiment(
            description="Try to find good settings for predicting MACE using "
                        "1 raw ECG and flat-features.",
            model=ecg_cnn,
            model_kwargs={
                'cnn_kwargs': hp.Choice([
                    {
                        'downsample': True,
                        'num_layers': num_layers,
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers),
                        'pool_size': hp.Choice(pool_sizes),
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
                    } for num_layers, pool_sizes in zip(
                        [2, 3, 4],
                        [range(7, 31), range(4, 11), range(3, 7)]
                    )
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
                    'ecgs': ['ecg_0'],
                    'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
                },
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
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
            'iterations': 400
        },
    )
    M_R2_FF_CNN_RS = HyperExperiment(
        template=Experiment(
            description="Try to find good settings for predicting MACE using "
                        "2 raw ECGs, with flat-features.",
            model=ecg_cnn,
            model_kwargs={
                'cnn_kwargs': hp.Choice([
                    {
                        'downsample': True,
                        'num_layers': num_layers,
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers),
                        'pool_size': hp.Choice(pool_sizes),
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
                    } for num_layers, pool_sizes in zip(
                        [2, 3, 4],
                        [range(7, 31), range(4, 11), range(3, 7)]
                    )
                ]),
                'cnn_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_ffnn_kwargs': hp.Choice([
                    None,
                    {
                        'sizes': hp.Choices([10, 20], k=1),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                        'batch_norms': [False]
                    }
                ]),
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
                    'ecgs': ['ecg_0', 'ecg_1'],
                    'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
                },
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
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
        random_seed=43,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )
    M_R1_CNN_RS = HyperExperiment(
        template=Experiment(
            description="Try to find good settings for predicting MACE using "
                        "1 raw ECG, without flat-features.",
            model=ecg_cnn,
            model_kwargs={
                'cnn_kwargs': hp.Choice([
                    {
                        'down_sample': True,
                        'num_layers': num_layers,
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers),
                        'pool_size': hp.Choice(pool_sizes),
                        'filter_first': hp.Int(8, 64, step=4),
                        'filter_last': hp.Int(8, 64, step=4),
                        'kernel_first': hp.Int(5, 65, step=4),
                        'kernel_last': hp.Int(5, 65, step=4),
                        'batch_norms': hp.Choices([True, False], k=num_layers),
                        'weight_decays': hp.Choices(
                            [1e-1, 1e-2, 1e-3, 0.0],
                            k=num_layers),
                    } for num_layers, pool_sizes in zip(
                        [2, 3, 4],
                        [range(7, 31), range(4, 11), range(3, 7)]
                    )
                ]),
                'ecg_ffnn_kwargs': hp.Choice([
                    None,
                    {
                        'sizes': hp.Choices([10, 50, 100], k=1),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                        'batch_norms': [False]
                    }
                ]),
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
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
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
        random_seed=44,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )
    M_R2_CNN_RS = HyperExperiment(
        template=Experiment(
            description="Try to find good settings for predicting MACE using "
                        "2 raw ECGs, without flat-features.",
            model=ecg_cnn,
            model_kwargs={
                'cnn_kwargs': hp.Choice([
                    {
                        'downsample': True,
                        'num_layers': num_layers,
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers),
                        'pool_size': hp.Choice(pool_sizes),
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
                    } for num_layers, pool_sizes in zip(
                        [2, 3, 4],
                        [range(7, 31), range(4, 11), range(3, 7)]
                    )
                ]),
                'cnn_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_ffnn_kwargs': hp.Choice([
                    None,
                    {
                        'sizes': hp.Choices([10, 20], k=1),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                        'batch_norms': [False]
                    }
                ]),
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
                    'ecgs': ['ecg_0', 'ecg_1']
                },
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
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
        random_seed=44,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )

    M_R1_RN_RS = HyperExperiment(
        template=Experiment(
            description="Random search for pretrained resnet using 1 ECG. "
                        "Search space is over ffnn parameters "
                        "following the flatten layer of the resnet. ",
            model=pretrained_resnet,
            model_kwargs={
                'freeze_resnet': False,
                'ecg_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [25, 50, 100, 200],
                            k=num_layers,
                            ascending=False,
                        ),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'batch_norms': hp.Choices(
                            [True, False],
                            k=num_layers
                        ),
                        'activity_regularizers': hp.Choices(
                            [0.01, 0.001, 0.0001, 0.00001, 0.0],
                            k=num_layers
                        ),
                        'kernel_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        ),
                        'bias_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2]
                ]),
                'ecg_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_comb_ffnn_kwargs': {
                    'sizes': hp.Choices([6, 10, 20], k=1),
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': None,
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0'],
                },
                'processing': {
                    'scale': 1000,
                    'ribeiro': True
                }
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': {
                        'scheduler': PiecewiseConstantDecay,
                        'scheduler_kwargs': {
                            'boundaries': [hp.Int(1525, 15250, step=305)],
                            'values': hp.SortedChoices(
                                [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
                                k=2, ascending=False
                            )
                        }
                    },
                }
            },
            epochs=100,
            batch_size=32,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1 / 3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=46,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )
    M_R2_RN_RS = HyperExperiment(
        template=Experiment(
            description="Random search for pretrained resnet using 2 ECGs. "
                        "Search space is over ffnn parameters "
                        "following the flatten layer of the resnet. ",
            model=pretrained_resnet,
            model_kwargs={
                'freeze_resnet': False,
                'ecg_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [25, 50, 100, 200],
                            k=num_layers,
                            ascending=False,
                        ),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'batch_norms': hp.Choices(
                            [True, False],
                            k=num_layers
                        ),
                        'activity_regularizers': hp.Choices(
                            [0.01, 0.001, 0.0001, 0.00001, 0.0],
                            k=num_layers
                        ),
                        'kernel_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        ),
                        'bias_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2]
                ]),
                'ecg_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_comb_ffnn_kwargs': {
                    'sizes': hp.Choices([6, 10, 20], k=1),
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': None,
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0', 'ecg_1'],
                },
                'processing': {
                    'scale': 1000,
                    'ribeiro': True
                }
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': {
                        'scheduler': PiecewiseConstantDecay,
                        'scheduler_kwargs': {
                            'boundaries': [hp.Int(1525, 15250, step=305)],
                            'values': hp.SortedChoices(
                                [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
                                k=2, ascending=False
                            )
                        }
                    },
                }
            },
            epochs=100,
            batch_size=32,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1 / 3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=47,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )
    M_R1_FF_RN_RS = HyperExperiment(
        template=Experiment(
            description="Random search for pretrained resnet using 1 ECG + "
                        "flat-features. Search space is over ffnn parameters "
                        "following the flatten layer of the resnet. ",
            model=pretrained_resnet,
            model_kwargs={
                'freeze_resnet': False,
                'ecg_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [25, 50, 100, 200],
                            k=num_layers,
                            ascending=False,
                        ),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'batch_norms': hp.Choices(
                            [True, False],
                            k=num_layers
                        ),
                        'activity_regularizers': hp.Choices(
                            [0.01, 0.001, 0.0001, 0.00001, 0.0],
                            k=num_layers
                        ),
                        'kernel_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        ),
                        'bias_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2]
                ]),
                'ecg_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_comb_ffnn_kwargs': {
                    'sizes': hp.Choices([6, 10, 20], k=1),
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': {
                    'sizes': [10],
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        k=1
                    ),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0'],
                    'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
                },
                'processing': {
                    'scale': 1000,
                    'ribeiro': True
                }
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': {
                        'scheduler': PiecewiseConstantDecay,
                        'scheduler_kwargs': {
                            'boundaries': [hp.Int(1525, 15250, step=305)],
                            'values': hp.SortedChoices(
                                [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
                                k=2, ascending=False
                            )
                        }
                    },
                }
            },
            epochs=100,
            batch_size=32,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1 / 3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=48,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )
    M_R2_FF_RN_RS = HyperExperiment(
        template=Experiment(
            description="Random search for pretrained resnet using 2 ECGs + "
                        "flat-features. Search space is over ffnn parameters "
                        "following the flatten layer of the resnet. ",
            model=pretrained_resnet,
            model_kwargs={
                'freeze_resnet': False,
                'ecg_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [25, 50, 100, 200],
                            k=num_layers,
                            ascending=False,
                        ),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'batch_norms': hp.Choices(
                            [True, False],
                            k=num_layers
                        ),
                        'activity_regularizers': hp.Choices(
                            [0.01, 0.001, 0.0001, 0.00001, 0.0],
                            k=num_layers
                        ),
                        'kernel_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        ),
                        'bias_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2]
                ]),
                'ecg_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_comb_ffnn_kwargs': {
                    'sizes': hp.Choices([6, 10, 20], k=1),
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': {
                    'sizes': [10],
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        k=1
                    ),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0', 'ecg_1'],
                    'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
                },
                'processing': {
                    'scale': 1000,
                    'ribeiro': True
                }
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': {
                        'scheduler': PiecewiseConstantDecay,
                        'scheduler_kwargs': {
                            'boundaries': [hp.Int(1525, 15250, step=305)],
                            'values': hp.SortedChoices(
                                [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
                                k=2, ascending=False
                            )
                        }
                    },
                }
            },
            epochs=100,
            batch_size=32,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1 / 3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=49,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )

    M_F1_NN_RS = HyperExperiment(
        template=Experiment(
            description="Random search over simple feed-forward neural "
                        "networks, using forberg features from 1 ECG",
            model=ffnn,
            model_kwargs={
                'ecg_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [25, 50, 100, 200],
                            k=num_layers,
                            ascending=False,
                        ),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'batch_norms': hp.Choices(
                            [True, False],
                            k=num_layers
                        ),
                        'activity_regularizers': hp.Choices(
                            [0.01, 0.001, 0.0001, 0.00001, 0.0],
                            k=num_layers
                        ),
                        'kernel_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        ),
                        'bias_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2]
                ]),
                'ecg_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_comb_ffnn_kwargs': {
                    'sizes': hp.Choices([6, 10, 20], k=1),
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        k=1
                    ),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': None
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'forberg': ['ecg_0']
                },
            },
            pre_processor=sklearn_process,
            pre_processor_kwargs={
                'forberg_ecg_0': {'processor': StandardScaler},
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
                }
            },
            epochs=100,
            batch_size=64,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1 / 3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=42,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )
    M_F2_NN_RS = HyperExperiment(
        template=Experiment(
            description="Random search over simple feed-forward neural "
                        "networks, using forberg features from 2 ECGs.",
            model=ffnn,
            model_kwargs={
                'ecg_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [25, 50, 100, 200],
                            k=num_layers,
                            ascending=False,
                        ),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'batch_norms': hp.Choices(
                            [True, False],
                            k=num_layers
                        ),
                        'activity_regularizers': hp.Choices(
                            [0.01, 0.001, 0.0001, 0.00001, 0.0],
                            k=num_layers
                        ),
                        'kernel_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        ),
                        'bias_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2]
                ]),
                'ecg_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_comb_ffnn_kwargs': {
                    'sizes': hp.Choices([6, 10, 20], k=1),
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        k=1
                    ),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': None,
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'forberg': ['ecg_0', 'ecg_1']
                },
            },
            pre_processor=sklearn_process,
            pre_processor_kwargs={
                'forberg_ecg_0': {'processor': StandardScaler},
                'forberg_ecg_1': {'processor': StandardScaler},
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
                }
            },
            epochs=100,
            batch_size=64,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1 / 3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=43,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )
    M_F1_FF_NN_RS = HyperExperiment(
        template=Experiment(
            description="Random search over simple feed-forward neural "
                        "networks, using forberg features from 1 ECG, and "
                        "flat-features. ",
            model=ffnn,
            model_kwargs={
                'ecg_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [25, 50, 100, 200],
                            k=num_layers,
                            ascending=False,
                        ),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'batch_norms': hp.Choices(
                            [True, False],
                            k=num_layers
                        ),
                        'activity_regularizers': hp.Choices(
                            [0.01, 0.001, 0.0001, 0.00001, 0.0],
                            k=num_layers
                        ),
                        'kernel_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        ),
                        'bias_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2]
                ]),
                'ecg_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_comb_ffnn_kwargs': {
                    'sizes': hp.Choices([6, 10, 20], k=1),
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        k=1
                    ),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': {
                    'sizes': [10],
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        k=1
                    ),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt'],
                    'forberg': ['ecg_0']
                },
            },
            pre_processor=sklearn_process,
            pre_processor_kwargs={
                'flat_features': {'processor': StandardScaler},
                'forberg_ecg_0': {'processor': StandardScaler},
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
                }
            },
            epochs=100,
            batch_size=64,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1 / 3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=44,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )
    M_F2_FF_NN_RS = HyperExperiment(
        template=Experiment(
            description="Random search over simple feed-forward neural "
                        "networks, using forberg features from 2 ECGs, and "
                        "flat-features. ",
            model=ffnn,
            model_kwargs={
                'ecg_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [25, 50, 100, 200],
                            k=num_layers,
                            ascending=False,
                        ),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'batch_norms': hp.Choices(
                            [True, False],
                            k=num_layers
                        ),
                        'activity_regularizers': hp.Choices(
                            [0.01, 0.001, 0.0001, 0.00001, 0.0],
                            k=num_layers
                        ),
                        'kernel_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        ),
                        'bias_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2]
                ]),
                'ecg_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_comb_ffnn_kwargs': {
                    'sizes': hp.Choices([6, 10, 20], k=1),
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        k=1
                    ),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': {
                    'sizes': [10],
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        k=1
                    ),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt'],
                    'forberg': ['ecg_0', 'ecg_1']
                },
            },
            pre_processor=sklearn_process,
            pre_processor_kwargs={
                'flat_features': {'processor': StandardScaler},
                'forberg_ecg_0': {'processor': StandardScaler},
                'forberg_ecg_1': {'processor': StandardScaler},
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
                }
            },
            epochs=100,
            batch_size=64,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1 / 3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=45,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )
