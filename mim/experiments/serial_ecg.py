from enum import Enum

from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

from mim.experiments.experiments import Experiment
from mim.extractors.esc_trop import EscTrop
from mim.models.simple_nn import ecg_cnn

from mim.cross_validation import ChronologicalSplit


class ECST(Experiment, Enum):
    M_R1_CNN1 = Experiment(
        description='Predicting MACE-30 using single raw ECG in a simple '
                    '2-layer CNN.',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'dropout': 0.3,
                'filter_first': 32,
                'filter_last': 32,
                'kernel_first': 16,
                'kernel_last': 16,
                'pool_size': 16,
                'batch_norm': True,
                'dense': False,
                'downsample': False
            },
            'dense_size': 10,
            'dropout': 0.3
        },
        epochs=200,
        batch_size=64,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-4}
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'beat',
                'ecgs': ['ecg_0']
            },
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R1_CNN2 = M_R1_CNN1._replace(
        description='Try adjusting the final dense-layer size from 10 to 100.',
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'dropout': 0.3,
                'filter_first': 32,
                'filter_last': 32,
                'kernel_first': 16,
                'kernel_last': 16,
                'pool_size': 16,
                'batch_norm': True,
                'dense': False,
                'downsample': True
            },
            'dense_size': 100,
            'dropout': 0.3
        }
    )
