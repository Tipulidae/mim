from enum import Enum

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from keras.optimizers import Adam

from mim.experiments.experiments import Experiment
from mim.models.util import CosineDecay
from projects.transfer.extractor import TargetTask, SourceTask
from projects.transfer.models import cnn, resnet_v2


class Target(Experiment, Enum):
    RN2_RAW_100 = Experiment(
        description='',
        model=resnet_v2,
        model_kwargs={},
        extractor=TargetTask,
        extractor_kwargs={
            'index': {'train_percent': 1.0},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': CosineDecay,
                    'scheduler_kwargs': {
                        'initial_learning_rate': 0.0,
                        'warmup_target': 5e-4,
                        'alpha': 1e-6,
                        # There seems to be 40 steps per epoch.
                        'warmup_steps': 10*40,
                        'decay_steps': 190*40,
                    }
                }
            }
        },
        epochs=200,
        batch_size=512,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        metrics=['auc'],
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_learning_rate=True,
    )
    RN2_RAW_90 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.9},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': CosineDecay,
                    'scheduler_kwargs': {
                        'initial_learning_rate': 0.0,
                        'warmup_target': 5e-4,
                        'alpha': 1e-6,
                        'warmup_steps': 10*36,
                        'decay_steps': 190*36,
                    }
                }
            }
        },
    )
    RN2_RAW_80 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.8},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': CosineDecay,
                    'scheduler_kwargs': {
                        'initial_learning_rate': 0.0,
                        'warmup_target': 5e-4,
                        'alpha': 1e-6,
                        'warmup_steps': 10*32,
                        'decay_steps': 190*32,
                    }
                }
            }
        },
    )
    RN2_RAW_70 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.7},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': CosineDecay,
                    'scheduler_kwargs': {
                        'initial_learning_rate': 0.0,
                        'warmup_target': 5e-4,
                        'alpha': 1e-6,
                        'warmup_steps': 10*28,
                        'decay_steps': 190*28,
                    }
                }
            }
        },
    )
    RN2_RAW_60 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.6},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': CosineDecay,
                    'scheduler_kwargs': {
                        'initial_learning_rate': 0.0,
                        'warmup_target': 5e-4,
                        'alpha': 1e-6,
                        'warmup_steps': 10*24,
                        'decay_steps': 190*24,
                    }
                }
            }
        },
    )
    RN2_RAW_50 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.5},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': CosineDecay,
                    'scheduler_kwargs': {
                        'initial_learning_rate': 0.0,
                        'warmup_target': 5e-4,
                        'alpha': 1e-6,
                        'warmup_steps': 10*20,
                        'decay_steps': 190*20,
                    }
                }
            }
        },
    )
    RN2_RAW_40 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.4},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': CosineDecay,
                    'scheduler_kwargs': {
                        'initial_learning_rate': 0.0,
                        'warmup_target': 5e-4,
                        'alpha': 1e-6,
                        'warmup_steps': 10*16,
                        'decay_steps': 190*16,
                    }
                }
            }
        },
    )
    RN2_RAW_30 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.3},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': CosineDecay,
                    'scheduler_kwargs': {
                        'initial_learning_rate': 0.0,
                        'warmup_target': 5e-4,
                        'alpha': 1e-6,
                        'warmup_steps': 10*12,
                        'decay_steps': 190*12,
                    }
                }
            }
        },
    )
    RN2_RAW_20 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.2},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': CosineDecay,
                    'scheduler_kwargs': {
                        'initial_learning_rate': 0.0,
                        'warmup_target': 5e-4,
                        'alpha': 1e-6,
                        'warmup_steps': 10*8,
                        'decay_steps': 190*8,
                    }
                }
            }
        },
    )
    RN2_RAW_10 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.1},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': CosineDecay,
                    'scheduler_kwargs': {
                        'initial_learning_rate': 0.0,
                        'warmup_target': 5e-4,
                        'alpha': 1e-6,
                        'warmup_steps': 10*4,
                        'decay_steps': 190*4,
                    }
                }
            }
        },
    )

    CNN1_R100 = Experiment(
        description='Uses M_R1_CNN1 from the serial ECGs project, which '
                    'performed well on the task of predicting MACE using only '
                    'the raw ECG signal.',
        model=cnn,
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'down_sample': False,
                'dropouts': [0.5, 0.4],
                'pool_size': 15,
                'filter_first': 28,
                'filter_last': 8,
                'kernel_first': 61,
                'kernel_last': 17,
                'batch_norms': [False, False],
                'weight_decays': [0.0, 0.01],
            },
            'ffnn_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False]
            },
        },
        extractor=TargetTask,
        extractor_kwargs={
            'index': {'train_percent': 1.0},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.0001}
        },
        epochs=100,
        batch_size=64,
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
    )
    CNN2_R100 = CNN1_R100._replace(
        description='Changes to the CosineDecay learning schedule and adds a '
                    'bit of regularization to the final dense layers.',
        epochs=200,
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'down_sample': False,
                'dropouts': [0.5, 0.4],
                'pool_size': 15,
                'filter_first': 28,
                'filter_last': 8,
                'kernel_first': 61,
                'kernel_last': 17,
                'batch_norms': [False, False],
                'weight_decays': [0.0, 0.01],
            },
            'ffnn_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False],
                'regularizer': 0.001
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': CosineDecay,
                    'scheduler_kwargs': {
                        'initial_learning_rate': 0.0,
                        'warmup_target': 1e-4,
                        'alpha': 3e-6,
                        'warmup_steps': 10*319,
                        'decay_steps': 200*319,
                    }
                }
            }
        },
        save_learning_rate=True,
    )


class Source(Experiment, Enum):
    RN2_R_SEX = Experiment(
        description='',
        model=resnet_v2,
        model_kwargs={},
        extractor=SourceTask,
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
            },
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': CosineDecay,
                    'scheduler_kwargs': {
                        'initial_learning_rate': 0.0,
                        'warmup_target': 5e-4,
                        'alpha': 1e-6,
                        'warmup_steps': 10*1628,
                        'decay_steps': 200*1628,
                    }
                }
            }
        },
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 0.95,
            'random_state': 515,
        },
        epochs=200,
        batch_size=512,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        metrics=['auc'],
        data_fits_in_memory=False,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_learning_rate=True,
        save_model_checkpoints={
            'save_best_only': False,
            'save_freq': 'epoch',
            'save_weights_only': True
        }
    )

    CNN2_R_SEX = Experiment(
        description='',
        extractor=SourceTask,
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
            },
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
        },
        model=cnn,
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'down_sample': False,
                'dropouts': [0.5, 0.4],
                'pool_size': 15,
                'filter_first': 28,
                'filter_last': 8,
                'kernel_first': 61,
                'kernel_last': 17,
                'batch_norms': [False, False],
                'weight_decays': [0.0, 0.01],
            },
            'ffnn_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False],
                'regularizer': 0.001
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': CosineDecay,
                    'scheduler_kwargs': {
                        'initial_learning_rate': 0.0,
                        'warmup_target': 1e-4,
                        'alpha': 3e-6,
                        'warmup_steps': 10*1628,
                        'decay_steps': 200*1628,
                    }
                }
            }
        },
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 0.95,
            'random_state': 515,
        },
        save_model_checkpoints={
            'save_best_only': False,
            'save_freq': 'epoch',
            'save_weights_only': True
        },
        epochs=200,
        batch_size=512,
        save_learning_rate=True,
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
    )
    CNN1_R5_F16_SEX = Experiment(
        description='Using only a random subset of the data',
        model=cnn,
        extractor=SourceTask,
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'small_subset': True,
            },
            'labels': {},
            'features': {
                'mode': 'raw',
                'precision': 16,
                'ribeiro': False,
            },
        },
        model_kwargs={
            'cnn_kwargs': {
                'initial_batch_norm': False,
                'num_layers': 2,
                'down_sample': False,
                'dropouts': [0.5, 0.4],
                'pool_size': 15,
                'filter_first': 28,
                'filter_last': 8,
                'kernel_first': 61,
                'kernel_last': 17,
                'batch_norms': [False, False],
                'weight_decays': [0.0, 0.01],
            },
            'ffnn_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False]
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.0001}
        },
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 0.7,
            'random_state': 515,
        },
        epochs=100,
        batch_size=64,
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        metrics=['accuracy', 'auc'],
        use_tensorboard=True,
    )
    CNN1_R5_F32_SEX = CNN1_R5_F16_SEX._replace(
        description='',
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'small_subset': True,
            },
            'labels': {},
            'features': {
                'mode': 'raw',
                'precision': 32,
                'ribeiro': False,
            },
        },
    )
    CNN1_R5_F64_SEX = CNN1_R5_F16_SEX._replace(
        description='',
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'small_subset': True,
            },
            'labels': {},
            'features': {
                'mode': 'raw',
                'precision': 64,
                'ribeiro': False,
            },
        },
    )
    CNN1_R5_F16_SEX_GENERATOR = CNN1_R5_F16_SEX._replace(
        description='',
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'small_subset': True,
            },
            'labels': {},
            'features': {
                'mode': 'raw',
                'precision': 16,
                'ribeiro': False,
                'fits_in_memory': False
            },
        },
    )
