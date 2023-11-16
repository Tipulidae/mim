from enum import Enum

from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import GroupShuffleSplit
from keras.optimizers import Adam

from mim.experiments.experiments import Experiment
from mim.models.util import CosineDecayWithWarmup
from projects.transfer.extractor import TargetTask, SourceTask
from projects.transfer.models import cnn, resnet_v1, resnet_v2, pretrained


class Target(Experiment, Enum):
    TEST = Experiment(
        description='Testing the new prediction history callback',
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
        epochs=10,
        batch_size=512,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        metrics=['auc'],
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True
    )
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
    RN1_R090 = RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.9},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True,
        },
    )
    RN1_R080 = RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.8},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True,
        },
    )
    RN1_R070 = RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.7},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True,
        },
    )
    RN1_R060 = RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.6},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True,
        },
    )
    RN1_R050 = RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.5},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True,
        },
    )
    RN1_R040 = RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.4},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True,
        },
    )
    RN1_R030 = RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.3},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True,
        },
    )
    RN1_R020 = RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.2},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True,
        },
    )
    RN1_R010 = RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.1},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True,
        },
    )

    RN2_R100 = Experiment(
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
    RN2_R090 = RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.9},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
    )
    RN2_R080 = RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.8},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
    )
    RN2_R070 = RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.7},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
    )
    RN2_R060 = RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.6},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
    )
    RN2_R050 = RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.5},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
    )
    RN2_R040 = RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.4},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
    )
    RN2_R030 = RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.3},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
    )
    RN2_R020 = RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.2},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
        },
    )
    RN2_R010 = RN2_R100._replace(
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
        optimizer=Adam,
        learning_rate=0.0001,
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

    PT_CNN1_R100 = Experiment(
        description='Uses CNN1 model pre-trained on sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'CNN1_R_SEX',
                'commit': '4bbad47036ea117143012a5089fc9b9e3f5d8956',
                'epoch': 200,
                'trainable': False,
                'final_layer_index': -6,
                'suffix': '_cnn1',
            },
            'final_mlp_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False]
            }
        },
        extractor=TargetTask,
        extractor_kwargs={
            'index': {'train_percent': 1.0},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
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
        batch_size=64,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
    )
    PT_CNN1_R090 = PT_CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.9},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
            'fits_in_memory': True
        },
    )
    PT_CNN1_R080 = PT_CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.8},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
            'fits_in_memory': True
        },
    )
    PT_CNN1_R070 = PT_CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.7},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
            'fits_in_memory': True
        },
    )
    PT_CNN1_R060 = PT_CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.6},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
            'fits_in_memory': True
        },
    )
    PT_CNN1_R050 = PT_CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.5},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
            'fits_in_memory': True
        },
    )
    PT_CNN1_R040 = PT_CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.4},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
            'fits_in_memory': True
        },
    )
    PT_CNN1_R030 = PT_CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.3},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
            'fits_in_memory': True
        },
    )
    PT_CNN1_R020 = PT_CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.2},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
            'fits_in_memory': True
        },
    )
    PT_CNN1_R010 = PT_CNN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.1},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': False},
            'fits_in_memory': True
        },
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
    PT_RN1_R090 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.9},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_R080 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.8},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_R070 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.7},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_R060 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.6},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_R050 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.5},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_R040 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.4},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_R030 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.3},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_R020 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.2},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_R010 = PT_RN1_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.1},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )

    PTA_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'dd82f8c595bcf1d3c0e915cffb1d55a008ae7ce2',
                'epoch': 100,
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
    PTA_RN1_R090 = PTA_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.9},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTA_RN1_R080 = PTA_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.8},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTA_RN1_R070 = PTA_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.7},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTA_RN1_R060 = PTA_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.6},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTA_RN1_R050 = PTA_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.5},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTA_RN1_R040 = PTA_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.4},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTA_RN1_R030 = PTA_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.3},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTA_RN1_R020 = PTA_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.2},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTA_RN1_R010 = PTA_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.1},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )

    PTAS_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age & sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE_SEX',
                'commit': '5eb3b1586ba3a808cf6237a4c73faa6d85eefc1a',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -3,
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
        save_val_pred_history=True,
    )
    PTAS_RN1_R090 = PTAS_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.9},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTAS_RN1_R080 = PTAS_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.8},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTAS_RN1_R070 = PTAS_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.7},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTAS_RN1_R060 = PTAS_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.6},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTAS_RN1_R050 = PTAS_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.5},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTAS_RN1_R040 = PTAS_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.4},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTAS_RN1_R030 = PTAS_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.3},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTAS_RN1_R020 = PTAS_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.2},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTAS_RN1_R010 = PTAS_RN1_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.1},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PTAS1000 = PTAS_RN1_R100._replace(
        epochs=1000
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
    PT_RN2_R090 = PT_RN2_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.9},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN2_R080 = PT_RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.8},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN2_R070 = PT_RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.7},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN2_R060 = PT_RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.6},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN2_R050 = PT_RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.5},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN2_R040 = PT_RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.4},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN2_R030 = PT_RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.3},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN2_R020 = PT_RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.2},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN2_R010 = PT_RN2_R100._replace(
        extractor_kwargs={
            'index': {'train_percent': 0.1},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )

    PT_RN1_SE8_R100 = Experiment(
        description='Uses RN1 model + SE block, trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_SE8_R100_SEX',
                'commit': 'ad81b5d5d8420d5192f4b634d910066683c55468',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_se8',
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
        save_val_pred_history=True
    )
    PT_RN1_SE8_R090 = PT_RN1_SE8_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.9},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_SE8_R080 = PT_RN1_SE8_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.8},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_SE8_R070 = PT_RN1_SE8_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.7},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_SE8_R060 = PT_RN1_SE8_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.6},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_SE8_R050 = PT_RN1_SE8_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.5},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_SE8_R040 = PT_RN1_SE8_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.4},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_SE8_R030 = PT_RN1_SE8_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.3},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_SE8_R020 = PT_RN1_SE8_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.2},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )
    PT_RN1_SE8_R010 = PT_RN1_SE8_R100._replace(
        description='',
        extractor_kwargs={
            'index': {'train_percent': 0.1},
            'labels': {},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': True
        },
    )


class Source(Experiment, Enum):
    RN1_R100_SEX = Experiment(
        description='A 4 block ResNet architecture from Ribeiro et al. '
                    'Trained here to predict sex using the raw ECG signal. '
                    'Adapted to use 8 input leads instead of 12.',
        model=resnet_v1,
        model_kwargs={},
        extractor=SourceTask,
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'train_percent': 1.0
            },
            'labels': {'sex': True, 'age': False},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': False
        },
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'steps_per_epoch': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 5e-4,
                'alpha': 1e-6,
                'warmup_epochs': 10,
                'decay_epochs': 90
            }
        },
        use_predefined_splits=True,
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
    RN1_R090_SEX = RN1_R100_SEX._replace(
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'train_percent': 0.9
            },
            'labels': {'sex': True, 'age': False},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': False
        },
    )
    RN1_R080_SEX = RN1_R100_SEX._replace(
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'train_percent': 0.8
            },
            'labels': {'sex': True, 'age': False},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': False
        },
    )
    RN1_R070_SEX = RN1_R100_SEX._replace(
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'train_percent': 0.7
            },
            'labels': {'sex': True, 'age': False},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': False
        },
    )
    RN1_R060_SEX = RN1_R100_SEX._replace(
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'train_percent': 0.6
            },
            'labels': {'sex': True, 'age': False},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': False
        },
    )
    RN1_R050_SEX = RN1_R100_SEX._replace(
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'train_percent': 0.5
            },
            'labels': {'sex': True, 'age': False},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': False
        },
    )
    RN1_R040_SEX = RN1_R100_SEX._replace(
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'train_percent': 0.4
            },
            'labels': {'sex': True, 'age': False},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': False
        },
    )
    RN1_R030_SEX = RN1_R100_SEX._replace(
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'train_percent': 0.3
            },
            'labels': {'sex': True, 'age': False},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': False
        },
    )
    RN1_R020_SEX = RN1_R100_SEX._replace(
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'train_percent': 0.2
            },
            'labels': {'sex': True, 'age': False},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': False
        },
    )
    RN1_R010_SEX = RN1_R100_SEX._replace(
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'train_percent': 0.1
            },
            'labels': {'sex': True, 'age': False},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': False
        },
    )
    RN2_R100_SEX = RN1_R100_SEX._replace(
        description='A 12 block ResNet architecture from Gustafsson et al. '
                    'Trained here to predict sex using the raw ECG signal.',
        model=resnet_v2,
    )
    RN1_SE4_R100_SEX = Experiment(
        description='Same as RN1, but with added Squeeze Excite (SE) layer.',
        model=resnet_v1,
        model_kwargs={
            'residual_kwargs': {
                'use_se_layer': True,
                'reduction_ratio': 4
            }
        },
        extractor=SourceTask,
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'train_percent': 1.0
            },
            'labels': {'sex': True, 'age': False},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': False
        },
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'steps_per_epoch': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 5e-4,
                'alpha': 1e-6,
                'warmup_epochs': 10,
                'decay_epochs': 90
            }
        },
        use_predefined_splits=True,
        epochs=100,
        batch_size=512,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        metrics=['auc', 'accuracy'],
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        save_model_checkpoints={
            'save_best_only': False,
            'save_freq': 'epoch',
            'save_weights_only': False
        }
    )
    RN1_SE8_R100_SEX = RN1_SE4_R100_SEX._replace(
        model_kwargs={
            'residual_kwargs': {
                'use_se_layer': True,
                'reduction_ratio': 8
            }
        },
    )
    RN1_SE16_R100_SEX = RN1_SE4_R100_SEX._replace(
        model_kwargs={
            'residual_kwargs': {
                'use_se_layer': True,
                'reduction_ratio': 16
            }
        },
    )
    RN1_SE24_R100_SEX = RN1_SE4_R100_SEX._replace(
        model_kwargs={
            'residual_kwargs': {
                'use_se_layer': True,
                'reduction_ratio': 24
            }
        },
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

    RN1_R100_AGE = Experiment(
        description='Predict age using the 4-block ResNet.',
        model=resnet_v1,
        model_kwargs={},
        extractor=SourceTask,
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'train_percent': 1.0
            },
            'labels': {'age': True, 'sex': False},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': False
        },
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'steps_per_epoch': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 5e-4,
                'alpha': 1e-6,
                'warmup_epochs': 10,
                'decay_epochs': 90
            }
        },
        use_predefined_splits=True,
        epochs=100,
        batch_size=512,
        loss='mean_absolute_error',
        scoring=r2_score,
        metrics=['mae', 'mse'],
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_learning_rate=True,
        save_model_checkpoints={
            'save_best_only': False,
            'save_freq': 'epoch',
            'save_weights_only': False
        }
    )

    RN1_R100_AGE_SEX = Experiment(
        description='Predict age and sex using the 4-block ResNet.',
        model=resnet_v1,
        model_kwargs={},
        extractor=SourceTask,
        extractor_kwargs={
            'index': {
                'exclude_train_aliases': True,
                'train_percent': 1.0
            },
            'labels': {'age': True, 'sex': True},
            'features': {'mode': 'raw', 'ribeiro': True},
            'fits_in_memory': False
        },
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'steps_per_epoch': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 5e-4,
                'alpha': 1e-6,
                'warmup_epochs': 10,
                'decay_epochs': 90
            }
        },
        use_predefined_splits=True,
        epochs=100,
        batch_size=512,
        loss={
            'sex': 'binary_crossentropy',
            'age': 'mean_absolute_error'
        },
        loss_weights={'sex': 1.0, 'age': 0.045},
        scoring=r2_score,
        metrics={
            'sex': ['acc', 'auc'],
            'age': ['mae', 'mse']
        },
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_learning_rate=True,
        save_model_checkpoints={
            'save_best_only': False,
            'save_freq': 'epoch',
            'save_weights_only': False
        }
    )
