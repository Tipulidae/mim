from enum import Enum

from sklearn.metrics import roc_auc_score
from tensorflow.keras.optimizers import Adam

from mim.experiments.experiments import Experiment
from projects.transfer.extractor import TargetTask
from projects.transfer.models import cnn, resnet_v2


class Target(Experiment, Enum):
    TEST = Experiment(
        description='',
        model=cnn,
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': False,
                'num_layers': 2,
                'dropout': 0.3,
                'filter_first': 32,
                'filter_last': 16,
                'kernel_first': 11,
                'kernel_last': 5,
            },
            'ffnn_kwargs': {
                'sizes': [10],
                'dropout': [0.3],
                'batch_norm': [False],
                'regularizer': [0.01]
            },
        },
        extractor=TargetTask,
        extractor_kwargs={
            'labels': {'sex': True},
            'features': {'ecg_mode': 'raw'},
            'index': {
                'train_percent': 0.01,
                'exclude_test_aliases': False
            }
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.001,
            }
        },
        epochs=200,
        batch_size=64,
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        use_predefined_splits=True,
        scoring=roc_auc_score,
        metrics=['auc']
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
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.0005,
            }
        },
        epochs=200,
        batch_size=512,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        metrics=['auc'],
        use_predefined_splits=True,
        building_model_requires_development_data=True,
    )
