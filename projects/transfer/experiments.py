from enum import Enum

from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import roc_auc_score
from tensorflow.keras.optimizers import Adam

from mim.experiments.experiments import Experiment
from projects.transfer.extractor import TargetTask
from projects.transfer.models import cnn


class Transfer(Experiment, Enum):
    BASELINE = Experiment(
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

    TEST = Experiment(
        description='',
        model=cnn,
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': False,
                'num_layers': 3,
                'dropouts': [0.2, 0.3, 0.4],
                'filters': [16, 16, 16],
                'kernels': [7, 7, 7],
                'weight_decays': [0.03, 0.01, 0.003],
                'pool_size': 5,
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
            'index': {'size_percent': 0.01}
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.0003,
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
