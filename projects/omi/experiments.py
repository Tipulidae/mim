from enum import Enum

from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from mim.experiments.experiments import Experiment
from projects.omi.extractor import OMIExtractor, UseAdditionalECGs
from projects.serial_ecgs.models import ecg_cnn


class ClassifyOMI(Experiment, Enum):
    TEMP = Experiment(
        description='foo',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'down_sample': True,
                'dropouts': [0.5, 0.4],
                'pool_size': 15,
                'filter_first': 28,
                'filter_last': 8,
                'kernel_first': 61,
                'kernel_last': 17,
                'batch_norms': [False, False],
                'weight_decays': [0.0, 0.01],
            },
            'ecg_ffnn_kwargs': {
                'sizes': [10],
                'dropout': [0.4],
                'batch_norm': [False]
            },
            'ecg_comb_ffnn_kwargs': None,
            'flat_ffnn_kwargs': None,
            # 'final_ffnn_kwargs': {
            #     'sizes': [100],
            #     'dropouts': [0.3],
            #     'batch_norms': [False]
            # }
        },
        extractor=OMIExtractor,
        extractor_kwargs={
            'labels': {
                'stenosis_limit': 90,
                'tnt_limit': 750
            }
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.0001}
        },
        epochs=100,
        batch_size=64,
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'test_size': 1 / 3,
            'n_splits': 1,
            'random_state': 8999
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    TEMP2 = TEMP._replace(
        extractor_kwargs={
            'labels': {
                'stenosis_limit': 100,
                'tnt_limit': 1000
            }
        },
    )
    TEMP_AUGMENT = TEMP._replace(
        augmentation=UseAdditionalECGs,
    )
