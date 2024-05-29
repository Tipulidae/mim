from enum import Enum

import torch
from keras.optimizers.legacy import Adam as LegacyAdam
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from mim.experiments.experiments import Experiment
from mim.models.util import cosine_decay_with_warmup_torch
from projects.omi.extractor import OMIExtractor, UseAdditionalECGs
from projects.serial_ecgs.models import ecg_cnn
from projects.transfer.models import pretrained_pt


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
        extractor_labels={
            'omi': True,
            'nomi': False,
            'ami': False
        },
        optimizer=LegacyAdam,
        learning_rate=1e-4,
        epochs=3,
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
    TEMP_AUGMENT = TEMP._replace(
        augmentation=UseAdditionalECGs,
    )
    XRN50_PTA_OMI = Experiment(
        description='XRN50-model pretrained on age.',
        model=pretrained_pt,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'XRN50A_R100_AGE',
                'commit': '661e83a57709b413ab7fac4907d60977c6cc0a8d',
                'epoch': 37,
                'trainable': False,
            }
        },
        extractor=OMIExtractor,
        extractor_labels={
            'omi': True,
            'nomi': False,
            'ami': False
        },
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'test_size': 1 / 3,
            'n_splits': 1,
            'random_state': 8999
        },
        building_model_requires_development_data=True,
        data_fits_in_memory=True,
        scoring=roc_auc_score,
        metrics=['auc'],
        loss=torch.nn.BCEWithLogitsLoss,
        optimizer=torch.optim.Adam,
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-4,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 30
            }
        },
        epochs=200,
        batch_size=256,
        unfreeze_after_epoch=40,
        use_tensorboard=True,
    )
    XRN50_PTA_AUGMENTED_OMI = XRN50_PTA_OMI._replace(
        description='XRN50, pretrained on age, predicting OMI. Training data '
                    'augmented with additional ECGs from the ED visit.',
        augmentation=UseAdditionalECGs,
    )
