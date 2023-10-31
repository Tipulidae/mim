from enum import Enum

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from keras.optimizers import Adam

from mim.experiments.experiments import Experiment
from mim.models.util import CosineDecayWithWarmup
from projects.transfer.extractor import TargetTask, SourceTask
from projects.transfer.models import cnn, resnet_v1, resnet_v2, pretrained


class Target(Experiment, Enum):
    RN1_R100 = Experiment(
        description='Training the ResNet v1 from scratch.',
        model=resnet_v1,
        model_kwargs={},
        extractor=TargetTask,
        extractor_kwargs={
            'index': {'train_percent': 1.0},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True,
        },
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 5e-4,
                'alpha': 1e-6,
                'warmup_epochs': 10,
                'decay_epochs': 90,
                'steps_per_epoch': -1
            }
        },
        epochs=100,
        batch_size=512,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        metrics=['auc'],
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_learning_rate=True,
    )
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
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 5e-4,
                'alpha': 1e-6,
                'warmup_epochs': 10,
                'decay_epochs': 190,
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
    )
    RN2_RAW_80 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.8},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
    )
    RN2_RAW_70 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.7},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
    )
    RN2_RAW_60 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.6},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
    )
    RN2_RAW_50 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.5},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
    )
    RN2_RAW_40 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.4},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
    )
    RN2_RAW_30 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.3},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
    )
    RN2_RAW_20 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.2},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
    )
    RN2_RAW_10 = RN2_RAW_100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.1},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
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
    CNN1_R090 = CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.9},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
        },
    )
    CNN1_R080 = CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.8},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
        },
    )
    CNN1_R070 = CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.7},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
        },
    )
    CNN1_R060 = CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.6},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
        },
    )
    CNN1_R050 = CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.5},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
        },
    )
    CNN1_R040 = CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.4},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
        },
    )
    CNN1_R030 = CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.3},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
        },
    )
    CNN1_R020 = CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.2},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
        },
    )
    CNN1_R010 = CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.1},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
        },
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
                    'scheduler': CosineDecayWithWarmup,
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

    PT_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R_SEX',
                'commit': '32c9a77ea6c7def6d0d78a31c547d47069c75606',
                'epoch': 59,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_kwargs={
            'index': {'train_percent': 1.0},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 30,
                'steps_per_epoch': -1
            }
        },
        epochs=200,
        batch_size=512,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
    )
    PT_RN1_R90 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.9},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_R80 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.8},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_R70 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.7},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_R60 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.6},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_R50 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.5},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_R40 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.4},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_R30 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.3},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_R20 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.2},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_R10 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.1},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )

    PT_RN2_R100 = Experiment(
        description='Uses RN2 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN2_R_SEX',
                'commit': 'bdaca8d1d5a01f38e80f139ba0afc9f9d4221512',
                'epoch': 40,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn2',
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            },
        },
        extractor=TargetTask,
        extractor_kwargs={
            'index': {'train_percent': 1.0},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 1e-4,
                'alpha': 0.1,
                'warmup_epochs': 10,
                'decay_epochs': 20,
                'steps_per_epoch': -1
            }
        },
        epochs=100,
        batch_size=512,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
    )
    PT_RN2_R90 = PT_RN2_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.9},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN2_R80 = PT_RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.8},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN2_R70 = PT_RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.7},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN2_R60 = PT_RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.6},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN2_R50 = PT_RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.5},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN2_R40 = PT_RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.4},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN2_R30 = PT_RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.3},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN2_R20 = PT_RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.2},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN2_R10 = PT_RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.1},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )


class Source(Experiment, Enum):
    RN1_R_SEX = Experiment(
        description='A 4 block ResNet architecture from Ribeiro et al. '
                    'Trained here to predict sex using the raw ECG signal. '
                    'Adapted to use 8 input leads instead of 12.',
        model=resnet_v1,
        model_kwargs={},
        extractor=SourceTask,
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
            },
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': False
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': CosineDecayWithWarmup,
                    'scheduler_kwargs': {
                        'initial_learning_rate': 0.0,
                        'warmup_target': 5e-4,
                        'alpha': 1e-6,
                        'warmup_steps': 10*1628,
                        'decay_steps': 100*1628,
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
        epochs=100,
        batch_size=512,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        metrics=['auc', 'accuracy'],
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_learning_rate=True,
        save_model_checkpoints={
            'save_best_only': False,
            'save_freq': 'epoch',
            'save_weights_only': False
        }
    )
    RN2_R_SEX = RN1_R_SEX._replace(
        description='A 12 block ResNet architecture from Gustafsson et al. '
                    'Trained here to predict sex using the raw ECG signal.',
        model=resnet_v2,
    )
    CNN1_R_SEX = Experiment(
        description='M_R1_CNN1 from the serial ECGs project. Predicting '
                    'sex using the raw ECG signal.',
        extractor=SourceTask,
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
            },
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
            'fits_in_memory': False
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
                'batch_norm': [False, False]
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': CosineDecayWithWarmup,
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
            'save_weights_only': False
        },
        epochs=200,
        batch_size=512,
        save_learning_rate=True,
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
    )
    CNN2_R_SEX = CNN1_R_SEX._replace(
        description='Same as CNN1, but with some extra regularization at the '
                    'final layer.',
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
    )
