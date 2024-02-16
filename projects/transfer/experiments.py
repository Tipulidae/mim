from enum import Enum

from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from keras.optimizers import Adam
from keras.optimizers.legacy import Adam as LegacyAdam
import torch

from mim.experiments.experiments import Experiment
from mim.experiments.extractor import sklearn_process
from mim.models.util import (
    CosineDecayWithWarmup, cosine_decay_with_warmup_torch
)
from projects.transfer.extractor import TargetTask, SourceTask
from projects.transfer.extractor import PTBXL as PTBXLExtractor
from projects.transfer.models import (
    cnn, resnet_v1, resnet_v2, pretrained, pretrained_pt, pretrained_parallel,
    ribeiros_resnet, xrn50
)


class Target(Experiment, Enum):
    # EXPERIMENTS USING NETWORKS TRAINED FROM SCRATCH:
    RN1_R100 = Experiment(
        description='Training the ResNet v1 from scratch.',
        model=resnet_v1,
        model_kwargs={},
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True}
        },
        data_fits_in_memory=True,
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 5e-4,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 90,
                'steps_per_epoch': -1
            }
        },
        epochs=100,
        batch_size=256,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        metrics=['auc'],
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    RN1_R090 = RN1_R100._replace(extractor_index={'train_percent': 0.9})
    RN1_R080 = RN1_R100._replace(extractor_index={'train_percent': 0.8})
    RN1_R070 = RN1_R100._replace(extractor_index={'train_percent': 0.7})
    RN1_R060 = RN1_R100._replace(extractor_index={'train_percent': 0.6})
    RN1_R050 = RN1_R100._replace(extractor_index={'train_percent': 0.5})
    RN1_R040 = RN1_R100._replace(extractor_index={'train_percent': 0.4})
    RN1_R030 = RN1_R100._replace(extractor_index={'train_percent': 0.3})
    RN1_R020 = RN1_R100._replace(extractor_index={'train_percent': 0.2})
    RN1_R010 = RN1_R100._replace(extractor_index={'train_percent': 0.1})

    RN2_R100 = Experiment(
        description='',
        model=resnet_v2,
        model_kwargs={},
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True}
        },
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 5e-4,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 190,
                'steps_per_epoch': -1
            }
        },
        epochs=200,
        batch_size=256,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        metrics=['auc'],
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    RN2_R090 = RN2_R100._replace(extractor_index={'train_percent': 0.9})
    RN2_R080 = RN2_R100._replace(extractor_index={'train_percent': 0.8})
    RN2_R070 = RN2_R100._replace(extractor_index={'train_percent': 0.7})
    RN2_R060 = RN2_R100._replace(extractor_index={'train_percent': 0.6})
    RN2_R050 = RN2_R100._replace(extractor_index={'train_percent': 0.5})
    RN2_R040 = RN2_R100._replace(extractor_index={'train_percent': 0.4})
    RN2_R030 = RN2_R100._replace(extractor_index={'train_percent': 0.3})
    RN2_R020 = RN2_R100._replace(extractor_index={'train_percent': 0.2})
    RN2_R010 = RN2_R100._replace(extractor_index={'train_percent': 0.1})

    RN1_ASR100 = Experiment(
        description='Training the ResNet v1 from scratch. '
                    'Uses ECG + age + sex',
        model=resnet_v1,
        model_kwargs={
            'flat_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.0
            },
            'ecg_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
            'flat_features': {'age': True, 'sex': True, 'scale_age': True}
        },
        data_fits_in_memory=True,
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 5e-4,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 90,
                'steps_per_epoch': -1
            }
        },
        epochs=100,
        batch_size=256,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        metrics=['auc'],
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True
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
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        optimizer=Adam,
        learning_rate=0.0001,
        epochs=200,
        batch_size=256,
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    CNN1_R090 = CNN1_R100._replace(extractor_index={'train_percent': 0.9})
    CNN1_R080 = CNN1_R100._replace(extractor_index={'train_percent': 0.8})
    CNN1_R070 = CNN1_R100._replace(extractor_index={'train_percent': 0.7})
    CNN1_R060 = CNN1_R100._replace(extractor_index={'train_percent': 0.6})
    CNN1_R050 = CNN1_R100._replace(extractor_index={'train_percent': 0.5})
    CNN1_R040 = CNN1_R100._replace(extractor_index={'train_percent': 0.4})
    CNN1_R030 = CNN1_R100._replace(extractor_index={'train_percent': 0.3})
    CNN1_R020 = CNN1_R100._replace(extractor_index={'train_percent': 0.2})
    CNN1_R010 = CNN1_R100._replace(extractor_index={'train_percent': 0.1})

    XRN50_R100 = Experiment(
        description='xresnet50 without augmentation, predicting AMI',
        model=xrn50,
        model_kwargs={
            'initial_bn': False,
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {
                'mode': 'raw',
                'ribeiro': False
            }
        },
        data_fits_in_memory=True,
        optimizer=torch.optim.Adam,
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-4,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 90
            }
        },
        epochs=100,
        batch_size=256,
        loss=torch.nn.BCEWithLogitsLoss,
        scoring=roc_auc_score,
        metrics=['auc'],
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_model=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_ami_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    XRN50_R090 = XRN50_R100._replace(extractor_index={'train_percent': 0.9})
    XRN50_R080 = XRN50_R100._replace(extractor_index={'train_percent': 0.8})
    XRN50_R070 = XRN50_R100._replace(extractor_index={'train_percent': 0.7})
    XRN50_R060 = XRN50_R100._replace(extractor_index={'train_percent': 0.6})
    XRN50_R050 = XRN50_R100._replace(extractor_index={'train_percent': 0.5})
    XRN50_R040 = XRN50_R100._replace(extractor_index={'train_percent': 0.4})
    XRN50_R030 = XRN50_R100._replace(extractor_index={'train_percent': 0.3})
    XRN50_R020 = XRN50_R100._replace(extractor_index={'train_percent': 0.2})
    XRN50_R010 = XRN50_R100._replace(extractor_index={'train_percent': 0.1})

    XRN50V2_R100 = XRN50_R100._replace(
        description='Adds batch sliding window augmentation with max '
                    'reduction.',
        model_kwargs={
            'initial_bn': False,
            'augmentation': {'mode': 'batch', 'reduction': 'max'}
        },
    )
    XRN50V3_R100 = XRN50_R100._replace(
        description='Adds batch sliding window augmentation with mean '
                    'reduction.',
        model_kwargs={
            'initial_bn': False,
            'augmentation': {'mode': 'batch', 'reduction': 'mean'}
        },
    )
    XRN50V4_R100 = XRN50_R100._replace(
        description='Reduced lr, increased epochs',
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-6,
                'warmup_target': 1e-5,
                'alpha': 0.01,
                'warmup_epochs': 30,
                'decay_epochs': 170
            }
        },
        epochs=200,
    )
    XRN50V5_R100 = XRN50_R100._replace(
        description='Adds 0.1 weight decay',
        optimizer_kwargs={'weight_decay': 0.1}
    )
    XRN50V6_R100 = XRN50_R100._replace(
        description='lr 1e-4, epochs 200',
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-5,
                'warmup_target': 1e-4,
                'alpha': 0.01,
                'warmup_epochs': 30,
                'decay_epochs': 170
            }
        },
        epochs=200,
    )
    XRN50V7_R100 = XRN50_R100._replace(
        description='Augmentation, lr 1e-4, epochs 200',
        model_kwargs={
            'initial_bn': False,
            'augmentation': {'mode': 'batch', 'reduction': 'mean'}
        },
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-5,
                'warmup_target': 1e-4,
                'alpha': 0.01,
                'warmup_epochs': 30,
                'decay_epochs': 170
            }
        },
        epochs=200,
    )

    XRN50A_R100 = Experiment(
        description='xresnet50 with augmentation, predicting AMI.',
        model=xrn50,
        model_kwargs={
            'initial_bn': False,
            'augmentation': {'mode': 'batch', 'reduction': 'mean'}
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {
                'mode': 'raw',
                'ribeiro': False
            }
        },
        data_fits_in_memory=True,
        optimizer=torch.optim.Adam,
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-4,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 90
            }
        },
        epochs=100,
        batch_size=256,
        loss=torch.nn.BCEWithLogitsLoss,
        scoring=roc_auc_score,
        metrics=['auc'],
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_model=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_ami_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    XRN50A_R090 = XRN50A_R100._replace(extractor_index={'train_percent': 0.9})
    XRN50A_R080 = XRN50A_R100._replace(extractor_index={'train_percent': 0.8})
    XRN50A_R070 = XRN50A_R100._replace(extractor_index={'train_percent': 0.7})
    XRN50A_R060 = XRN50A_R100._replace(extractor_index={'train_percent': 0.6})
    XRN50A_R050 = XRN50A_R100._replace(extractor_index={'train_percent': 0.5})
    XRN50A_R040 = XRN50A_R100._replace(extractor_index={'train_percent': 0.4})
    XRN50A_R030 = XRN50A_R100._replace(extractor_index={'train_percent': 0.3})
    XRN50A_R020 = XRN50A_R100._replace(extractor_index={'train_percent': 0.2})
    XRN50A_R010 = XRN50A_R100._replace(extractor_index={'train_percent': 0.1})

    # EXPERIMENTS USING PRE-TRAINED NETWORKS:
    PTS100_CNN1_R100 = Experiment(
        description='Uses CNN1 model pre-trained on sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'CNN1_R100_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 200,
                'trainable': False,
                'final_layer_index': -6,
                'suffix': '_cnn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False]
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS100_CNN1_R090 = PTS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS100_CNN1_R080 = PTS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS100_CNN1_R070 = PTS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS100_CNN1_R060 = PTS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS100_CNN1_R050 = PTS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS100_CNN1_R040 = PTS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS100_CNN1_R030 = PTS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS100_CNN1_R020 = PTS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS100_CNN1_R010 = PTS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS090_CNN1_R100 = Experiment(
        description='Uses CNN1 model pre-trained on sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'CNN1_R090_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 200,
                'trainable': False,
                'final_layer_index': -6,
                'suffix': '_cnn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False]
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS090_CNN1_R090 = PTS090_CNN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS090_CNN1_R080 = PTS090_CNN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS090_CNN1_R070 = PTS090_CNN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS090_CNN1_R060 = PTS090_CNN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS090_CNN1_R050 = PTS090_CNN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS090_CNN1_R040 = PTS090_CNN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS090_CNN1_R030 = PTS090_CNN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS090_CNN1_R020 = PTS090_CNN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS090_CNN1_R010 = PTS090_CNN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS080_CNN1_R100 = Experiment(
        description='Uses CNN1 model pre-trained on sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'CNN1_R080_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 200,
                'trainable': False,
                'final_layer_index': -6,
                'suffix': '_cnn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False]
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS080_CNN1_R090 = PTS080_CNN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS080_CNN1_R080 = PTS080_CNN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS080_CNN1_R070 = PTS080_CNN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS080_CNN1_R060 = PTS080_CNN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS080_CNN1_R050 = PTS080_CNN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS080_CNN1_R040 = PTS080_CNN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS080_CNN1_R030 = PTS080_CNN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS080_CNN1_R020 = PTS080_CNN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS080_CNN1_R010 = PTS080_CNN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS070_CNN1_R100 = Experiment(
        description='Uses CNN1 model pre-trained on sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'CNN1_R070_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 200,
                'trainable': False,
                'final_layer_index': -6,
                'suffix': '_cnn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False]
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS070_CNN1_R090 = PTS070_CNN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS070_CNN1_R080 = PTS070_CNN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS070_CNN1_R070 = PTS070_CNN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS070_CNN1_R060 = PTS070_CNN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS070_CNN1_R050 = PTS070_CNN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS070_CNN1_R040 = PTS070_CNN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS070_CNN1_R030 = PTS070_CNN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS070_CNN1_R020 = PTS070_CNN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS070_CNN1_R010 = PTS070_CNN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS060_CNN1_R100 = Experiment(
        description='Uses CNN1 model pre-trained on sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'CNN1_R010_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 200,
                'trainable': False,
                'final_layer_index': -6,
                'suffix': '_cnn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False]
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS060_CNN1_R090 = PTS060_CNN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS060_CNN1_R080 = PTS060_CNN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS060_CNN1_R070 = PTS060_CNN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS060_CNN1_R060 = PTS060_CNN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS060_CNN1_R050 = PTS060_CNN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS060_CNN1_R040 = PTS060_CNN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS060_CNN1_R030 = PTS060_CNN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS060_CNN1_R020 = PTS060_CNN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS060_CNN1_R010 = PTS060_CNN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS050_CNN1_R100 = Experiment(
        description='Uses CNN1 model pre-trained on sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'CNN1_R050_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 200,
                'trainable': False,
                'final_layer_index': -6,
                'suffix': '_cnn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False]
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS050_CNN1_R090 = PTS050_CNN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS050_CNN1_R080 = PTS050_CNN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS050_CNN1_R070 = PTS050_CNN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS050_CNN1_R060 = PTS050_CNN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS050_CNN1_R050 = PTS050_CNN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS050_CNN1_R040 = PTS050_CNN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS050_CNN1_R030 = PTS050_CNN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS050_CNN1_R020 = PTS050_CNN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS050_CNN1_R010 = PTS050_CNN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS040_CNN1_R100 = Experiment(
        description='Uses CNN1 model pre-trained on sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'CNN1_R040_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 200,
                'trainable': False,
                'final_layer_index': -6,
                'suffix': '_cnn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False]
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS040_CNN1_R090 = PTS040_CNN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS040_CNN1_R080 = PTS040_CNN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS040_CNN1_R070 = PTS040_CNN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS040_CNN1_R060 = PTS040_CNN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS040_CNN1_R050 = PTS040_CNN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS040_CNN1_R040 = PTS040_CNN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS040_CNN1_R030 = PTS040_CNN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS040_CNN1_R020 = PTS040_CNN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS040_CNN1_R010 = PTS040_CNN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS030_CNN1_R100 = Experiment(
        description='Uses CNN1 model pre-trained on sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'CNN1_R030_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 200,
                'trainable': False,
                'final_layer_index': -6,
                'suffix': '_cnn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False]
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS030_CNN1_R090 = PTS030_CNN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS030_CNN1_R080 = PTS030_CNN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS030_CNN1_R070 = PTS030_CNN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS030_CNN1_R060 = PTS030_CNN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS030_CNN1_R050 = PTS030_CNN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS030_CNN1_R040 = PTS030_CNN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS030_CNN1_R030 = PTS030_CNN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS030_CNN1_R020 = PTS030_CNN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS030_CNN1_R010 = PTS030_CNN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS020_CNN1_R100 = Experiment(
        description='Uses CNN1 model pre-trained on sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'CNN1_R020_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 200,
                'trainable': False,
                'final_layer_index': -6,
                'suffix': '_cnn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False]
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS020_CNN1_R090 = PTS020_CNN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS020_CNN1_R080 = PTS020_CNN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS020_CNN1_R070 = PTS020_CNN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS020_CNN1_R060 = PTS020_CNN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS020_CNN1_R050 = PTS020_CNN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS020_CNN1_R040 = PTS020_CNN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS020_CNN1_R030 = PTS020_CNN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS020_CNN1_R020 = PTS020_CNN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS020_CNN1_R010 = PTS020_CNN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS010_CNN1_R100 = Experiment(
        description='Uses CNN1 model pre-trained on sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'CNN1_R010_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 200,
                'trainable': False,
                'final_layer_index': -6,
                'suffix': '_cnn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False]
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS010_CNN1_R090 = PTS010_CNN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS010_CNN1_R080 = PTS010_CNN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS010_CNN1_R070 = PTS010_CNN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS010_CNN1_R060 = PTS010_CNN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS010_CNN1_R050 = PTS010_CNN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS010_CNN1_R040 = PTS010_CNN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS010_CNN1_R030 = PTS010_CNN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS010_CNN1_R020 = PTS010_CNN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS010_CNN1_R010 = PTS010_CNN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA100_CNN1_R100 = Experiment(
        description='Uses CNN1 model pre-trained on age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'CNN1_R100_AGE',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 200,
                'trainable': False,
                'final_layer_index': -6,
                'suffix': '_cnn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False]
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA100_CNN1_R090 = PTA100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA100_CNN1_R080 = PTA100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA100_CNN1_R070 = PTA100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA100_CNN1_R060 = PTA100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA100_CNN1_R050 = PTA100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA100_CNN1_R040 = PTA100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA100_CNN1_R030 = PTA100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA100_CNN1_R020 = PTA100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA100_CNN1_R010 = PTA100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTAS100_CNN1_R100 = Experiment(
        description='Uses CNN1 model pre-trained on age and sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'CNN1_R100_AGE_SEX',
                'commit': '05ee85195684bba26db810ed61cbd7830bc09e95',
                'epoch': 200,
                'trainable': False,
                'final_layer_index': -7,
                'suffix': '_cnn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [10, 100],
                'dropout': [0.4, 0.3],
                'batch_norm': [False, False]
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTAS100_CNN1_R090 = PTAS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTAS100_CNN1_R080 = PTAS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTAS100_CNN1_R070 = PTAS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTAS100_CNN1_R060 = PTAS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTAS100_CNN1_R050 = PTAS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTAS100_CNN1_R040 = PTAS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTAS100_CNN1_R030 = PTAS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTAS100_CNN1_R020 = PTAS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTAS100_CNN1_R010 = PTAS100_CNN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA100_XRN50_R100_V1 = Experiment(
        description='XRN50 pretrained on age',
        model=pretrained_pt,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'XRN50V1_R100_AGE',
                'commit': '',
                'epoch': 26,
                'trainable': False,
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        data_fits_in_memory=True,
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
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss=torch.nn.BCEWithLogitsLoss,
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_ami_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA100_XRN50_R100_V2 = PTA100_XRN50_R100_V1._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'XRN50V2_R100_AGE',
                'commit': '',
                'epoch': 47,
                'trainable': False,
            }
        },
    )
    PTA100_XRN50_R100_V3 = PTA100_XRN50_R100_V1._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'XRN50V2_R100_AGE',
                'commit': '',
                'epoch': 47,
                'trainable': False,
            }
        },
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 3e-5,
                'warmup_target': 3e-4,
                'alpha': 0.01,
                'warmup_epochs': 20,
                'decay_epochs': 60
            }
        },
        epochs=200,
        unfreeze_after_epoch=80,
    )
    PTA100_XRN50_R100_V4 = PTA100_XRN50_R100_V2._replace(
        description='Augmentation + 500 epochs',
        epochs=500
    )
    PTA100_XRN50_R100_V5 = PTA100_XRN50_R100_V1._replace(
        description='No augmentation + load later epoch (100)',
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'XRN50V1_R100_AGE',
                'commit': '',
                'epoch': 100,
                'trainable': False,
            }
        },
    )
    PTA100_XRN50_R100_V6 = PTA100_XRN50_R100_V2._replace(
        description='Augmentation + increased lr ',
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 3e-4,
                'warmup_target': 3e-3,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 30
            }
        },
    )

    PTA100_XRN50A_R100 = Experiment(
        description='Augmented XRN50 pretrained on age.',
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
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        data_fits_in_memory=True,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss=torch.nn.BCEWithLogitsLoss,
        scoring=roc_auc_score,
        metrics=['auc'],
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_ami_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA100_XRN50A_R090 = PTA100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA100_XRN50A_R080 = PTA100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA100_XRN50A_R070 = PTA100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA100_XRN50A_R060 = PTA100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA100_XRN50A_R050 = PTA100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA100_XRN50A_R040 = PTA100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA100_XRN50A_R030 = PTA100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA100_XRN50A_R020 = PTA100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA100_XRN50A_R010 = PTA100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS100_XRN50A_R100 = Experiment(
        description='Augmented XRN50 pretrained on sex.',
        model=pretrained_pt,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'XRN50A_R100_SEX',
                'commit': '661e83a57709b413ab7fac4907d60977c6cc0a8d',
                'epoch': 25,
                'trainable': False,
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        data_fits_in_memory=True,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss=torch.nn.BCEWithLogitsLoss,
        scoring=roc_auc_score,
        metrics=['auc'],
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_ami_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS100_XRN50A_R090 = PTS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS100_XRN50A_R080 = PTS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS100_XRN50A_R070 = PTS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS100_XRN50A_R060 = PTS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS100_XRN50A_R050 = PTS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS100_XRN50A_R040 = PTS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS100_XRN50A_R030 = PTS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS100_XRN50A_R020 = PTS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS100_XRN50A_R010 = PTS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTAS100_XRN50A_R100 = Experiment(
        description='Augmented XRN50 pretrained on age + sex.',
        model=pretrained_pt,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'XRN50A_R100_AGE_SEX',
                'commit': '661e83a57709b413ab7fac4907d60977c6cc0a8d',
                'epoch': 29,
                'trainable': False,
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': False},
        },
        data_fits_in_memory=True,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss=torch.nn.BCEWithLogitsLoss,
        scoring=roc_auc_score,
        metrics=['auc'],
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_ami_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTAS100_XRN50A_R090 = PTAS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTAS100_XRN50A_R080 = PTAS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTAS100_XRN50A_R070 = PTAS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTAS100_XRN50A_R060 = PTAS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTAS100_XRN50A_R050 = PTAS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTAS100_XRN50A_R040 = PTAS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTAS100_XRN50A_R030 = PTAS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTAS100_XRN50A_R020 = PTAS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTAS100_XRN50A_R010 = PTAS100_XRN50A_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS100_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS100_RN1_R090 = PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS100_RN1_R080 = PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS100_RN1_R070 = PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS100_RN1_R060 = PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS100_RN1_R050 = PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS100_RN1_R040 = PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS100_RN1_R030 = PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS100_RN1_R020 = PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS100_RN1_R010 = PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS090_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R090_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS090_RN1_R090 = PTS090_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS090_RN1_R080 = PTS090_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS090_RN1_R070 = PTS090_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS090_RN1_R060 = PTS090_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS090_RN1_R050 = PTS090_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS090_RN1_R040 = PTS090_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS090_RN1_R030 = PTS090_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS090_RN1_R020 = PTS090_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS090_RN1_R010 = PTS090_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS080_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R080_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS080_RN1_R090 = PTS080_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS080_RN1_R080 = PTS080_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS080_RN1_R070 = PTS080_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS080_RN1_R060 = PTS080_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS080_RN1_R050 = PTS080_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS080_RN1_R040 = PTS080_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS080_RN1_R030 = PTS080_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS080_RN1_R020 = PTS080_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS080_RN1_R010 = PTS080_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS070_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R070_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS070_RN1_R090 = PTS070_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS070_RN1_R080 = PTS070_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS070_RN1_R070 = PTS070_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS070_RN1_R060 = PTS070_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS070_RN1_R050 = PTS070_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS070_RN1_R040 = PTS070_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS070_RN1_R030 = PTS070_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS070_RN1_R020 = PTS070_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS070_RN1_R010 = PTS070_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS060_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R060_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS060_RN1_R090 = PTS060_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS060_RN1_R080 = PTS060_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS060_RN1_R070 = PTS060_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS060_RN1_R060 = PTS060_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS060_RN1_R050 = PTS060_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS060_RN1_R040 = PTS060_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS060_RN1_R030 = PTS060_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS060_RN1_R020 = PTS060_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS060_RN1_R010 = PTS060_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS050_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R050_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS050_RN1_R090 = PTS050_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS050_RN1_R080 = PTS050_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS050_RN1_R070 = PTS050_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS050_RN1_R060 = PTS050_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS050_RN1_R050 = PTS050_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS050_RN1_R040 = PTS050_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS050_RN1_R030 = PTS050_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS050_RN1_R020 = PTS050_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS050_RN1_R010 = PTS050_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS040_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R040_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS040_RN1_R090 = PTS040_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS040_RN1_R080 = PTS040_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS040_RN1_R070 = PTS040_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS040_RN1_R060 = PTS040_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS040_RN1_R050 = PTS040_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS040_RN1_R040 = PTS040_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS040_RN1_R030 = PTS040_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS040_RN1_R020 = PTS040_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS040_RN1_R010 = PTS040_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS030_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R030_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS030_RN1_R090 = PTS030_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS030_RN1_R080 = PTS030_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS030_RN1_R070 = PTS030_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS030_RN1_R060 = PTS030_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS030_RN1_R050 = PTS030_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS030_RN1_R040 = PTS030_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS030_RN1_R030 = PTS030_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS030_RN1_R020 = PTS030_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS030_RN1_R010 = PTS030_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS020_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R020_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS020_RN1_R090 = PTS020_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS020_RN1_R080 = PTS020_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS020_RN1_R070 = PTS020_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS020_RN1_R060 = PTS020_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS020_RN1_R050 = PTS020_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS020_RN1_R040 = PTS020_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS020_RN1_R030 = PTS020_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS020_RN1_R020 = PTS020_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS020_RN1_R010 = PTS020_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS010_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R010_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS010_RN1_R090 = PTS010_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS010_RN1_R080 = PTS010_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS010_RN1_R070 = PTS010_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS010_RN1_R060 = PTS010_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS010_RN1_R050 = PTS010_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS010_RN1_R040 = PTS010_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS010_RN1_R030 = PTS010_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS010_RN1_R020 = PTS010_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS010_RN1_R010 = PTS010_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS008_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R008_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS008_RN1_R090 = PTS008_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS008_RN1_R080 = PTS008_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS008_RN1_R070 = PTS008_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS008_RN1_R060 = PTS008_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS008_RN1_R050 = PTS008_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS008_RN1_R040 = PTS008_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS008_RN1_R030 = PTS008_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS008_RN1_R020 = PTS008_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS008_RN1_R010 = PTS008_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS006_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R006_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS006_RN1_R090 = PTS006_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS006_RN1_R080 = PTS006_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS006_RN1_R070 = PTS006_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS006_RN1_R060 = PTS006_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS006_RN1_R050 = PTS006_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS006_RN1_R040 = PTS006_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS006_RN1_R030 = PTS006_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS006_RN1_R020 = PTS006_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS006_RN1_R010 = PTS006_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS004_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R004_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS004_RN1_R090 = PTS004_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS004_RN1_R080 = PTS004_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS004_RN1_R070 = PTS004_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS004_RN1_R060 = PTS004_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS004_RN1_R050 = PTS004_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS004_RN1_R040 = PTS004_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS004_RN1_R030 = PTS004_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS004_RN1_R020 = PTS004_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS004_RN1_R010 = PTS004_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS002_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R002_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS002_RN1_R090 = PTS002_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS002_RN1_R080 = PTS002_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS002_RN1_R070 = PTS002_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS002_RN1_R060 = PTS002_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS002_RN1_R050 = PTS002_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS002_RN1_R040 = PTS002_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS002_RN1_R030 = PTS002_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS002_RN1_R020 = PTS002_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS002_RN1_R010 = PTS002_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA100_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA100_RN1_R090 = PTA100_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA100_RN1_R080 = PTA100_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA100_RN1_R070 = PTA100_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA100_RN1_R060 = PTA100_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA100_RN1_R050 = PTA100_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA100_RN1_R040 = PTA100_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA100_RN1_R030 = PTA100_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA100_RN1_R020 = PTA100_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA100_RN1_R010 = PTA100_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA090_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R090_AGE',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA090_RN1_R090 = PTA090_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA090_RN1_R080 = PTA090_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA090_RN1_R070 = PTA090_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA090_RN1_R060 = PTA090_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA090_RN1_R050 = PTA090_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA090_RN1_R040 = PTA090_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA090_RN1_R030 = PTA090_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA090_RN1_R020 = PTA090_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA090_RN1_R010 = PTA090_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA080_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R080_AGE',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA080_RN1_R090 = PTA080_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA080_RN1_R080 = PTA080_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA080_RN1_R070 = PTA080_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA080_RN1_R060 = PTA080_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA080_RN1_R050 = PTA080_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA080_RN1_R040 = PTA080_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA080_RN1_R030 = PTA080_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA080_RN1_R020 = PTA080_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA080_RN1_R010 = PTA080_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA070_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R070_AGE',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA070_RN1_R090 = PTA070_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA070_RN1_R080 = PTA070_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA070_RN1_R070 = PTA070_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA070_RN1_R060 = PTA070_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA070_RN1_R050 = PTA070_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA070_RN1_R040 = PTA070_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA070_RN1_R030 = PTA070_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA070_RN1_R020 = PTA070_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA070_RN1_R010 = PTA070_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA060_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R060_AGE',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA060_RN1_R090 = PTA060_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA060_RN1_R080 = PTA060_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA060_RN1_R070 = PTA060_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA060_RN1_R060 = PTA060_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA060_RN1_R050 = PTA060_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA060_RN1_R040 = PTA060_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA060_RN1_R030 = PTA060_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA060_RN1_R020 = PTA060_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA060_RN1_R010 = PTA060_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA050_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R050_AGE',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA050_RN1_R090 = PTA050_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA050_RN1_R080 = PTA050_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA050_RN1_R070 = PTA050_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA050_RN1_R060 = PTA050_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA050_RN1_R050 = PTA050_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA050_RN1_R040 = PTA050_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA050_RN1_R030 = PTA050_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA050_RN1_R020 = PTA050_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA050_RN1_R010 = PTA050_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA040_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R040_AGE',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA040_RN1_R090 = PTA040_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA040_RN1_R080 = PTA040_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA040_RN1_R070 = PTA040_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA040_RN1_R060 = PTA040_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA040_RN1_R050 = PTA040_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA040_RN1_R040 = PTA040_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA040_RN1_R030 = PTA040_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA040_RN1_R020 = PTA040_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA040_RN1_R010 = PTA040_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA030_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R030_AGE',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA030_RN1_R090 = PTA030_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA030_RN1_R080 = PTA030_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA030_RN1_R070 = PTA030_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA030_RN1_R060 = PTA030_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA030_RN1_R050 = PTA030_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA030_RN1_R040 = PTA030_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA030_RN1_R030 = PTA030_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA030_RN1_R020 = PTA030_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA030_RN1_R010 = PTA030_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA020_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R020_AGE',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA020_RN1_R090 = PTA020_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA020_RN1_R080 = PTA020_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA020_RN1_R070 = PTA020_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA020_RN1_R060 = PTA020_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA020_RN1_R050 = PTA020_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA020_RN1_R040 = PTA020_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA020_RN1_R030 = PTA020_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA020_RN1_R020 = PTA020_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA020_RN1_R010 = PTA020_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA010_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R010_AGE',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA010_RN1_R090 = PTA010_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA010_RN1_R080 = PTA010_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA010_RN1_R070 = PTA010_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA010_RN1_R060 = PTA010_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA010_RN1_R050 = PTA010_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA010_RN1_R040 = PTA010_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA010_RN1_R030 = PTA010_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA010_RN1_R020 = PTA010_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA010_RN1_R010 = PTA010_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA008_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R008_AGE',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA008_RN1_R090 = PTA008_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA008_RN1_R080 = PTA008_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA008_RN1_R070 = PTA008_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA008_RN1_R060 = PTA008_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA008_RN1_R050 = PTA008_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA008_RN1_R040 = PTA008_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA008_RN1_R030 = PTA008_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA008_RN1_R020 = PTA008_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA008_RN1_R010 = PTA008_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA006_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R006_AGE',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA006_RN1_R090 = PTA006_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA006_RN1_R080 = PTA006_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA006_RN1_R070 = PTA006_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA006_RN1_R060 = PTA006_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA006_RN1_R050 = PTA006_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA006_RN1_R040 = PTA006_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA006_RN1_R030 = PTA006_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA006_RN1_R020 = PTA006_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA006_RN1_R010 = PTA006_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA004_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R004_AGE',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA004_RN1_R090 = PTA004_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA004_RN1_R080 = PTA004_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA004_RN1_R070 = PTA004_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA004_RN1_R060 = PTA004_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA004_RN1_R050 = PTA004_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA004_RN1_R040 = PTA004_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA004_RN1_R030 = PTA004_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA004_RN1_R020 = PTA004_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA004_RN1_R010 = PTA004_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA002_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R002_AGE',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA002_RN1_R090 = PTA002_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA002_RN1_R080 = PTA002_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA002_RN1_R070 = PTA002_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA002_RN1_R060 = PTA002_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA002_RN1_R050 = PTA002_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA002_RN1_R040 = PTA002_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA002_RN1_R030 = PTA002_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA002_RN1_R020 = PTA002_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA002_RN1_R010 = PTA002_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTAS100_RN1_R100 = Experiment(
        description='Uses RN1 model trained to predict age & sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -3,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTAS100_RN1_R090 = PTAS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTAS100_RN1_R080 = PTAS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTAS100_RN1_R070 = PTAS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTAS100_RN1_R060 = PTAS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTAS100_RN1_R050 = PTAS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTAS100_RN1_R040 = PTAS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTAS100_RN1_R030 = PTAS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTAS100_RN1_R020 = PTAS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTAS100_RN1_R010 = PTAS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS100_RN1_ASR100 = Experiment(
        description='RN1 pre-trained on 100% of the data to predict sex, '
                    'used to predict AMI with ECG, age and sex as inputs.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            },
            'flat_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.0
            },
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
            'flat_features': {'age': True, 'sex': True, 'scale_age': True}
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS100_RN1_ASR090 = PTS100_RN1_ASR100._replace(
        extractor_index={'train_percent': 0.9})
    PTS100_RN1_ASR080 = PTS100_RN1_ASR100._replace(
        extractor_index={'train_percent': 0.8})
    PTS100_RN1_ASR070 = PTS100_RN1_ASR100._replace(
        extractor_index={'train_percent': 0.7})
    PTS100_RN1_ASR060 = PTS100_RN1_ASR100._replace(
        extractor_index={'train_percent': 0.6})
    PTS100_RN1_ASR050 = PTS100_RN1_ASR100._replace(
        extractor_index={'train_percent': 0.5})
    PTS100_RN1_ASR040 = PTS100_RN1_ASR100._replace(
        extractor_index={'train_percent': 0.4})
    PTS100_RN1_ASR030 = PTS100_RN1_ASR100._replace(
        extractor_index={'train_percent': 0.3})
    PTS100_RN1_ASR020 = PTS100_RN1_ASR100._replace(
        extractor_index={'train_percent': 0.2})
    PTS100_RN1_ASR010 = PTS100_RN1_ASR100._replace(
        extractor_index={'train_percent': 0.1})

    PTA100_PTS100_RN1_R100 = Experiment(
        description='Uses two pre-trained RN1 models in parallel to predict '
                    'AMI. One model is pre-trained on age, the other on sex.'
                    'batch-size reduced to fit model in memory.',
        model=pretrained_parallel,
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA100_PTS100_RN1_R090 = PTA100_PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA100_PTS100_RN1_R080 = PTA100_PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA100_PTS100_RN1_R070 = PTA100_PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA100_PTS100_RN1_R060 = PTA100_PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA100_PTS100_RN1_R050 = PTA100_PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA100_PTS100_RN1_R040 = PTA100_PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA100_PTS100_RN1_R030 = PTA100_PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA100_PTS100_RN1_R020 = PTA100_PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA100_PTS100_RN1_R010 = PTA100_PTS100_RN1_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTS100_RN2_R100 = Experiment(
        description='Uses RN2 model trained to predict sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN2_R100_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn2',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            },
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTS100_RN2_R090 = PTS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTS100_RN2_R080 = PTS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTS100_RN2_R070 = PTS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTS100_RN2_R060 = PTS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTS100_RN2_R050 = PTS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTS100_RN2_R040 = PTS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTS100_RN2_R030 = PTS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTS100_RN2_R020 = PTS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTS100_RN2_R010 = PTS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTA100_RN2_R100 = Experiment(
        description='Uses RN2 model trained to predict age.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN2_R100_AGE',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn2',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTA100_RN2_R090 = PTA100_RN2_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTA100_RN2_R080 = PTA100_RN2_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTA100_RN2_R070 = PTA100_RN2_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTA100_RN2_R060 = PTA100_RN2_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTA100_RN2_R050 = PTA100_RN2_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTA100_RN2_R040 = PTA100_RN2_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTA100_RN2_R030 = PTA100_RN2_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTA100_RN2_R020 = PTA100_RN2_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTA100_RN2_R010 = PTA100_RN2_R100._replace(
        extractor_index={'train_percent': 0.1})

    PTAS100_RN2_R100 = Experiment(
        description='Uses RN2 model trained to predict age & sex.',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN2_R100_AGE_SEX',
                'commit': '064e5e0e9c763ab3f53fed34d6484a6d79c02e8c',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -3,
                'suffix': '_rn2',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
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
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
            'monitor': 'val_real_auc',
            'mode': 'max',
            'save_best_only': True
        },
        test_model='best'
    )
    PTAS100_RN2_R090 = PTAS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.9})
    PTAS100_RN2_R080 = PTAS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.8})
    PTAS100_RN2_R070 = PTAS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.7})
    PTAS100_RN2_R060 = PTAS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.6})
    PTAS100_RN2_R050 = PTAS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.5})
    PTAS100_RN2_R040 = PTAS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.4})
    PTAS100_RN2_R030 = PTAS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.3})
    PTAS100_RN2_R020 = PTAS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.2})
    PTAS100_RN2_R010 = PTAS100_RN2_R100._replace(
        extractor_index={'train_percent': 0.1})

    PT_RIBEIRO_R100 = Experiment(
        description='Uses the network pretrained by Ribeiro on 2M ECGs.',
        model=ribeiros_resnet,
        model_kwargs={
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {
                'mode': 'raw',
                'ribeiro': False,
                'original_ribeiro': True,
            }
        },
        data_fits_in_memory=True,
        optimizer=LegacyAdam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 1e-2,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 30,
                'steps_per_epoch': -1
            }
        },
        epochs=400,
        batch_size=256,
        unfreeze_after_epoch=40,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc'],
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True
    )
    PT_RIBEIRO_R090 = PT_RIBEIRO_R100._replace(
        extractor_index={'train_percent': 0.9})
    PT_RIBEIRO_R080 = PT_RIBEIRO_R100._replace(
        extractor_index={'train_percent': 0.8})
    PT_RIBEIRO_R070 = PT_RIBEIRO_R100._replace(
        extractor_index={'train_percent': 0.7})
    PT_RIBEIRO_R060 = PT_RIBEIRO_R100._replace(
        extractor_index={'train_percent': 0.6})
    PT_RIBEIRO_R050 = PT_RIBEIRO_R100._replace(
        extractor_index={'train_percent': 0.5})
    PT_RIBEIRO_R040 = PT_RIBEIRO_R100._replace(
        extractor_index={'train_percent': 0.4})
    PT_RIBEIRO_R030 = PT_RIBEIRO_R100._replace(
        extractor_index={'train_percent': 0.3})
    PT_RIBEIRO_R020 = PT_RIBEIRO_R100._replace(
        extractor_index={'train_percent': 0.2})
    PT_RIBEIRO_R010 = PT_RIBEIRO_R100._replace(
        extractor_index={'train_percent': 0.1})

    LR_ASR100 = Experiment(
        description='Predict AMI with only age and sex',
        model=LogisticRegression,
        model_kwargs={
            'class_weight': 'balanced',
            'random_state': 123,
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'flat_features': {'age': True, 'sex': True, 'scale_age': True}
        },
        data_fits_in_memory=True,
        use_predefined_splits=True,
        scoring=roc_auc_score,
    )
    LR_PTA100_RN1_R100 = Experiment(
        description='Uses the age predictions from the RN1 network as input '
                    'to predict AMI.',
        model=LogisticRegression,
        model_kwargs={
            'class_weight': 'balanced',
            'random_state': 123,
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        extractor_processing={
            'process_with_xps': [
                {
                    'xp_project': 'transfer',
                    'xp_base': 'Source',
                    'xp_name': 'RN1_R100_AGE',
                    'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                    'epoch': 100,
                    'input_key': 'ecg'
                }
            ]
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'RN1_R100_AGE': {
                'processor': StandardScaler
            }
        },
        data_fits_in_memory=True,
        use_predefined_splits=True,
        scoring=roc_auc_score,
    )
    LR_PTS100_RN1_R100 = LR_PTA100_RN1_R100._replace(
        description='Uses the sex predictions from the RN1 network as input '
                    'to predict AMI.',
        extractor_processing={
            'process_with_xps': [
                {
                    'xp_project': 'transfer',
                    'xp_base': 'Source',
                    'xp_name': 'RN1_R100_SEX',
                    'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                    'epoch': 100,
                    'input_key': 'ecg'
                }
            ]
        },
        pre_processor=None
    )
    LR_PTA100_RN1_PTS100_RN1_R100 = LR_PTA100_RN1_R100._replace(
        description='Uses age prediction from RN1 and sex prediction from '
                    'another RN1 to predict AMI.',
        extractor_processing={
            'process_with_xps': [
                {
                    'xp_project': 'transfer',
                    'xp_base': 'Source',
                    'xp_name': 'RN1_R100_AGE',
                    'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                    'epoch': 100,
                    'input_key': 'ecg'
                },
                {
                    'xp_project': 'transfer',
                    'xp_base': 'Source',
                    'xp_name': 'RN1_R100_SEX',
                    'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                    'epoch': 100,
                    'input_key': 'ecg'
                }
            ]
        },
        pre_processor_kwargs={
            'RN1_R100_AGE': {
                'processor': StandardScaler
            }
        },
    )
    LR_PTA100_RN1_PTS100_RN1_ASR100 = LR_PTA100_RN1_PTS100_RN1_R100._replace(
        description='Uses the age & sex predictions from the RN1 network as '
                    'input to predict AMI. Adds the real age and sex as well.',
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
            'flat_features': {'age': True, 'sex': True, 'scale_age': True}
        },
    )


class TargetGridSearch(Experiment, Enum):
    TEST = Experiment(
        description='sanity check',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 20,
                'decay_epochs': 80,
                'steps_per_epoch': -1
            }
        },
        epochs=500,
        batch_size=512,
        unfreeze_after_epoch=100,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True
    )

    A100_D3_FC50_L20_D0 = Experiment(
        description='',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 20,
                'decay_epochs': 80,
                'steps_per_epoch': -1
            }
        },
        epochs=500,
        batch_size=512,
        unfreeze_after_epoch=100,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True
    )
    A100_D3_FC50_L20_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D3_FC50_L21_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D3_FC50_L21_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D3_FC50_L22_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D3_FC50_L22_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D3_FC100_L20_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D3_FC100_L20_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D3_FC100_L21_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D3_FC100_L21_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D3_FC100_L22_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D3_FC100_L22_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D3_FC200_L20_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D3_FC200_L20_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D3_FC200_L21_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D3_FC200_L21_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D3_FC200_L22_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D3_FC200_L22_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC50_L20_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC50_L20_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC50_L21_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC50_L21_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC50_L22_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC50_L22_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC100_L20_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC100_L20_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC100_L21_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC100_L21_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC100_L22_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC100_L22_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC200_L20_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC200_L20_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC200_L21_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC200_L21_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC200_L22_D0 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_D6_FC200_L22_D3 = A100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )

    S100_D3_FC50_L20_D0 = Experiment(
        description='',
        model=pretrained,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 20,
                'decay_epochs': 80,
                'steps_per_epoch': -1
            }
        },
        epochs=500,
        batch_size=512,
        unfreeze_after_epoch=100,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True
    )
    S100_D3_FC50_L20_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D3_FC50_L21_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D3_FC50_L21_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D3_FC50_L22_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D3_FC50_L22_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D3_FC100_L20_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D3_FC100_L20_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D3_FC100_L21_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D3_FC100_L21_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D3_FC100_L22_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D3_FC100_L22_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D3_FC200_L20_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D3_FC200_L20_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D3_FC200_L21_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D3_FC200_L21_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D3_FC200_L22_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D3_FC200_L22_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC50_L20_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC50_L20_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC50_L21_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC50_L21_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC50_L22_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC50_L22_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC100_L20_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC100_L20_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC100_L21_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC100_L21_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC100_L22_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC100_L22_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC200_L20_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC200_L20_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC200_L21_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC200_L21_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC200_L22_D0 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    S100_D6_FC200_L22_D3 = S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )

    A100_S100_D3_FC50_L20_D0 = Experiment(
        description='',
        model=pretrained_parallel,
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
        extractor=TargetTask,
        extractor_index={'train_percent': 1.0},
        extractor_features={
            'ecg_features': {'mode': 'raw', 'ribeiro': True},
        },
        data_fits_in_memory=True,
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 20,
                'decay_epochs': 80,
                'steps_per_epoch': -1
            }
        },
        epochs=500,
        batch_size=256,
        unfreeze_after_epoch=100,
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True
    )
    A100_S100_D3_FC50_L20_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D3_FC50_L21_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D3_FC50_L21_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D3_FC50_L22_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D3_FC50_L22_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D3_FC100_L20_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D3_FC100_L20_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D3_FC100_L21_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D3_FC100_L21_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D3_FC100_L22_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D3_FC100_L22_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D3_FC200_L20_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D3_FC200_L20_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D3_FC200_L21_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D3_FC200_L21_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D3_FC200_L22_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D3_FC200_L22_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.3,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC50_L20_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC50_L20_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC50_L21_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC50_L21_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC50_L22_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC50_L22_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [50],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC100_L20_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC100_L20_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC100_L21_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC100_L21_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC100_L22_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC100_L22_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [100],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC200_L20_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC200_L20_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 0.0, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC200_L21_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC200_L21_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-4, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC200_L22_D0 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.0,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )
    A100_S100_D6_FC200_L22_D3 = A100_S100_D3_FC50_L20_D0._replace(
        model_kwargs={
            'from_xp1': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_AGE',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_age',
                'input_key': 'ecg'
            },
            'from_xp2': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'RN1_R100_SEX',
                'commit': 'b5c829281bb845ff5d810a9de370a9512ea548b5',
                'epoch': 100,
                'trainable': False,
                'final_layer_index': -2,
                'suffix': '_rn1_sex',
                'input_key': 'ecg'
            },
            'ecg_dropout': 0.3,
            'final_mlp_kwargs': {
                'sizes': [200],
                'dropout': 0.6,
                'regularizer': {'activity': 1e-3, 'kernel': 0.0, 'bias': 0.0}
            }
        },
    )


class Source(Experiment, Enum):
    CNN1_R100_SEX = Experiment(
        description='M_R1_CNN1 from the serial ECGs project. Predicting '
                    'sex using the raw ECG signal.',
        extractor=SourceTask,
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 1.0
        },
        extractor_labels={'sex': True, 'age': False},
        extractor_features={'mode': 'raw', 'ribeiro': False},
        data_fits_in_memory=False,
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
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'steps_per_epoch': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 1e-4,
                'alpha': 3e-6,
                'warmup_epochs': 10,
                'decay_epochs': 200,
            }
        },
        use_predefined_splits=True,
        model_checkpoints={
            'save_best_only': False,
            'save_freq': 'epoch',
            'save_weights_only': False
        },
        epochs=200,
        batch_size=256,
        save_learning_rate=True,
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        use_tensorboard=True,
        save_val_pred_history=True
    )
    CNN1_R090_SEX = CNN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.9
        },
    )
    CNN1_R080_SEX = CNN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.8
        },
    )
    CNN1_R070_SEX = CNN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.7
        },
    )
    CNN1_R060_SEX = CNN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.6
        },
    )
    CNN1_R050_SEX = CNN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.5
        },
    )
    CNN1_R040_SEX = CNN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.4
        },
    )
    CNN1_R030_SEX = CNN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.3
        },
    )
    CNN1_R020_SEX = CNN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.2
        },
    )
    CNN1_R010_SEX = CNN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.1
        },
    )

    CNN1_LR_TEST = CNN1_R100_SEX._replace(
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'steps_per_epoch': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 90,
            }
        },
        epochs=100
    )
    CNN1_LR_TEST2 = CNN1_R100_SEX._replace(
        learning_rate=3e-5,
        epochs=200
    )
    CNN1_LR_TEST3 = CNN1_R100_SEX._replace(
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'steps_per_epoch': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 1e-4,
                'alpha': 1e-2,
                'warmup_epochs': 10,
                'decay_epochs': 490,
            }
        },
        epochs=500
    )

    CNN1_R100_AGE = Experiment(
        description='M_R1_CNN1 from the serial ECGs project. Predicting '
                    'age using the raw ECG signal.',
        extractor=SourceTask,
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 1.0
        },
        extractor_labels={'sex': False, 'age': True},
        extractor_features={'mode': 'raw', 'ribeiro': False},
        data_fits_in_memory=False,
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
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'steps_per_epoch': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 1e-4,
                'alpha': 3e-6,
                'warmup_epochs': 10,
                'decay_epochs': 200,
            }
        },
        use_predefined_splits=True,
        model_checkpoints={
            'save_best_only': False,
            'save_freq': 'epoch',
            'save_weights_only': False
        },
        epochs=200,
        batch_size=256,
        save_learning_rate=True,
        building_model_requires_development_data=True,
        loss='mean_absolute_error',
        scoring=r2_score,
        metrics=['mae', 'mse'],
        use_tensorboard=True,
        save_val_pred_history=True
    )
    CNN1_R090_AGE = CNN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.9
        },
    )
    CNN1_R080_AGE = CNN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.8
        },
    )
    CNN1_R070_AGE = CNN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.7
        },
    )
    CNN1_R060_AGE = CNN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.6
        },
    )
    CNN1_R050_AGE = CNN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.5
        },
    )
    CNN1_R040_AGE = CNN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.4
        },
    )
    CNN1_R030_AGE = CNN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.3
        },
    )
    CNN1_R020_AGE = CNN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.2
        },
    )
    CNN1_R010_AGE = CNN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.1
        },
    )
    CNN1_R008_AGE = CNN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.08
        },
    )
    CNN1_R006_AGE = CNN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.06
        },
    )
    CNN1_R004_AGE = CNN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.04
        },
    )
    CNN1_R002_AGE = CNN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.02
        },
    )

    CNN1_R100_AGE_SEX = Experiment(
        description='M_R1_CNN1 from the serial ECGs project. Predicting '
                    'age and sex using the raw ECG signal.',
        extractor=SourceTask,
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 1.0
        },
        extractor_labels={'sex': True, 'age': True},
        extractor_features={'mode': 'raw', 'ribeiro': False},
        data_fits_in_memory=False,
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
        optimizer=Adam,
        learning_rate={
            'scheduler': CosineDecayWithWarmup,
            'kwargs': {
                'decay_steps': -1,
                'steps_per_epoch': -1,
                'initial_learning_rate': 0.0,
                'warmup_target': 1e-4,
                'alpha': 3e-6,
                'warmup_epochs': 10,
                'decay_epochs': 200,
            }
        },
        use_predefined_splits=True,
        model_checkpoints={
            'save_best_only': False,
            'save_freq': 'epoch',
            'save_weights_only': False
        },
        epochs=200,
        batch_size=256,
        save_learning_rate=True,
        building_model_requires_development_data=True,
        loss={
            'sex': 'binary_crossentropy',
            'age': 'mean_absolute_error'
        },
        loss_weights={'sex': 1.0, 'age': 0.045},
        metrics={
            'sex': ['acc', 'auc'],
            'age': ['mae', 'mse']
        },
        scoring=r2_score,
        use_tensorboard=True,
        save_val_pred_history=True
    )

    XRN50V1_R100_AGE = Experiment(
        description='Using the xresnet50 architecture to predict age.',
        model=xrn50,
        model_kwargs={
            'initial_bn': False,
        },
        extractor=SourceTask,
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 1.0
        },
        extractor_labels={'sex': False, 'age': True},
        extractor_features={'mode': 'raw', 'ribeiro': False},
        data_fits_in_memory=False,
        optimizer=torch.optim.Adam,
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-4,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 90
            }
        },
        epochs=100,
        batch_size=256,
        loss=torch.nn.L1Loss,
        scoring=r2_score,
        metrics=['r2'],
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_model=True,
        model_checkpoints=True,
        save_learning_rate=True,
        save_val_pred_history=True
    )
    XRN50V2_R100_AGE = XRN50V1_R100_AGE._replace(
        description='Using the xresnet50 architecture to predict age. With '
                    'batch-wise sliding window augmentation with mean '
                    'reduction.',
        model_kwargs={
            'initial_bn': False,
            'augmentation': {'mode': 'batch', 'reduction': 'mean'}
        },
    )
    XRN50V3_R100_AGE = XRN50V1_R100_AGE._replace(
        description='Reduced learning rate to 1e-4.',
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-5,
                'warmup_target': 1e-4,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 90
            }
        },
    )
    XRN50V4_R100_AGE = XRN50V1_R100_AGE._replace(
        description='Weight decay 1e-3',
        optimizer_kwargs={'weight_decay': 1e-3}
    )

    XRN50_R100_AGE = Experiment(
        description='Unaugmented XRN50, v1.',
        model=xrn50,
        model_kwargs={
            'initial_bn': False,
        },
        extractor=SourceTask,
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 1.0
        },
        extractor_labels={'sex': False, 'age': True},
        extractor_features={'mode': 'raw', 'ribeiro': False},
        data_fits_in_memory=False,
        optimizer=torch.optim.Adam,
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-4,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 90
            }
        },
        epochs=100,
        batch_size=256,
        loss=torch.nn.L1Loss,
        scoring=r2_score,
        metrics=['r2'],
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_model=True,
        model_checkpoints=True,
        save_learning_rate=True,
        save_val_pred_history=True
    )
    XRN50_R100_SEX = Experiment(
        description='Unaugmented XRN50, v1. Predicting sex.',
        model=xrn50,
        model_kwargs={
            'initial_bn': False,
        },
        extractor=SourceTask,
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 1.0
        },
        extractor_labels={'sex': True, 'age': False},
        extractor_features={'mode': 'raw', 'ribeiro': False},
        data_fits_in_memory=False,
        optimizer=torch.optim.Adam,
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-4,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 90
            }
        },
        epochs=100,
        batch_size=256,
        loss=torch.nn.BCEWithLogitsLoss,
        scoring=roc_auc_score,
        metrics=['auc'],
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_model=True,
        model_checkpoints=True,
        save_learning_rate=True,
        save_val_pred_history=True
    )
    XRN50_R100_AGE_SEX = Experiment(
        description='Unaugmented XRN50, v1. Predicting age & sex.',
        model=xrn50,
        model_kwargs={
            'initial_bn': False,
        },
        extractor=SourceTask,
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 1.0
        },
        extractor_labels={'sex': True, 'age': True},
        extractor_features={'mode': 'raw', 'ribeiro': False},
        data_fits_in_memory=False,
        optimizer=torch.optim.Adam,
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-4,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 90
            }
        },
        epochs=100,
        batch_size=256,
        loss={
            'sex': torch.nn.BCEWithLogitsLoss,
            'age': torch.nn.L1Loss
        },
        loss_weights={'sex': 1.0, 'age': 0.045},
        scoring=None,
        metrics={
            'sex': ['auc'],
            'age': ['mae', 'r2']
        },
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_model=True,
        model_checkpoints=True,
        save_learning_rate=True,
        save_val_pred_history=True
    )

    XRN50A_R100_AGE = Experiment(
        description='Augmented XRN50, v1.',
        model=xrn50,
        model_kwargs={
            'initial_bn': False,
            'augmentation': {'mode': 'batch', 'reduction': 'mean'}
        },
        extractor=SourceTask,
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 1.0
        },
        extractor_labels={'sex': False, 'age': True},
        extractor_features={'mode': 'raw', 'ribeiro': False},
        data_fits_in_memory=False,
        optimizer=torch.optim.Adam,
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-4,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 90
            }
        },
        epochs=100,
        batch_size=256,
        loss=torch.nn.L1Loss,
        scoring=r2_score,
        metrics=['r2'],
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_model=True,
        model_checkpoints={'save_freq': 'epoch'},
        save_learning_rate=True,
        save_val_pred_history=True
    )
    XRN50A_R090_AGE = XRN50A_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.9
        },
    )
    XRN50A_R080_AGE = XRN50A_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.8
        },
    )
    XRN50A_R070_AGE = XRN50A_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.7
        },
    )
    XRN50A_R060_AGE = XRN50A_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.6
        },
    )
    XRN50A_R050_AGE = XRN50A_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.5
        },
    )
    XRN50A_R040_AGE = XRN50A_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.4
        },
    )
    XRN50A_R030_AGE = XRN50A_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.3
        },
    )
    XRN50A_R020_AGE = XRN50A_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.2
        },
    )
    XRN50A_R010_AGE = XRN50A_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.1
        },
    )
    XRN50A_R008_AGE = XRN50A_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.08
        },
    )
    XRN50A_R006_AGE = XRN50A_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.06
        },
    )
    XRN50A_R004_AGE = XRN50A_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.04
        },
    )
    XRN50A_R002_AGE = XRN50A_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.02
        },
    )

    XRN50A_R100_SEX = Experiment(
        description='Augmented XRN50, v1. Predicting sex.',
        model=xrn50,
        model_kwargs={
            'initial_bn': False,
            'augmentation': {'mode': 'batch', 'reduction': 'mean'}
        },
        extractor=SourceTask,
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 1.0
        },
        extractor_labels={'sex': True, 'age': False},
        extractor_features={'mode': 'raw', 'ribeiro': False},
        data_fits_in_memory=False,
        optimizer=torch.optim.Adam,
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-4,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 90
            }
        },
        epochs=100,
        batch_size=256,
        loss=torch.nn.BCEWithLogitsLoss,
        scoring=roc_auc_score,
        metrics=['auc'],
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_model=True,
        model_checkpoints={'save_freq': 'epoch'},
        save_learning_rate=True,
        save_val_pred_history=True
    )
    XRN50A_R100_AGE_SEX = Experiment(
        description='Augmented XRN50, v1. Predicting age & sex.',
        model=xrn50,
        model_kwargs={
            'initial_bn': False,
            'augmentation': {'mode': 'batch', 'reduction': 'mean'}
        },
        extractor=SourceTask,
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 1.0
        },
        extractor_labels={'sex': True, 'age': True},
        extractor_features={'mode': 'raw', 'ribeiro': False},
        data_fits_in_memory=False,
        optimizer=torch.optim.Adam,
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-4,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 90
            }
        },
        epochs=100,
        batch_size=256,
        loss={
            'sex': torch.nn.BCEWithLogitsLoss,
            'age': torch.nn.L1Loss
        },
        loss_weights={'sex': 1.0, 'age': 0.045},
        scoring=None,
        metrics={
            'sex': ['auc'],
            'age': ['mae', 'r2']
        },
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_model=True,
        model_checkpoints={'save_freq': 'epoch'},
        save_learning_rate=True,
        save_val_pred_history=True
    )

    RN1_R100_SEX = Experiment(
        description='A 4 block ResNet architecture from Ribeiro et al. '
                    'Trained here to predict sex using the raw ECG signal. '
                    'Adapted to use 8 input leads instead of 12.',
        model=resnet_v1,
        model_kwargs={},
        extractor=SourceTask,
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 1.0
        },
        extractor_labels={'sex': True, 'age': False},
        extractor_features={'mode': 'raw', 'ribeiro': True},
        data_fits_in_memory=False,
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
        batch_size=256,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        metrics=['auc', 'accuracy'],
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_learning_rate=True,
        model_checkpoints={
            'save_best_only': False,
            'save_freq': 'epoch',
            'save_weights_only': False
        },
        save_val_pred_history=True
    )
    RN1_R090_SEX = RN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.9
        },
    )
    RN1_R080_SEX = RN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.8
        },
    )
    RN1_R070_SEX = RN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.7
        },
    )
    RN1_R060_SEX = RN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.6
        },
    )
    RN1_R050_SEX = RN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.5
        },
    )
    RN1_R040_SEX = RN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.4
        },
    )
    RN1_R030_SEX = RN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.3
        },
    )
    RN1_R020_SEX = RN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.2
        },
    )
    RN1_R010_SEX = RN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.1
        },
    )
    RN1_R008_SEX = RN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.08
        },
    )
    RN1_R006_SEX = RN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.06
        },
    )
    RN1_R004_SEX = RN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.04
        },
    )
    RN1_R002_SEX = RN1_R100_SEX._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.02
        },
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
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 1.0
        },
        extractor_labels={'sex': True, 'age': False},
        extractor_features={'mode': 'raw', 'ribeiro': True},
        data_fits_in_memory=False,
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
        batch_size=256,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        metrics=['auc', 'accuracy'],
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        model_checkpoints={
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

    RN1_R100_AGE = Experiment(
        description='Predict age using the 4-block ResNet.',
        model=resnet_v1,
        model_kwargs={},
        extractor=SourceTask,
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 1.0
        },
        extractor_labels={'age': True, 'sex': False},
        extractor_features={'mode': 'raw', 'ribeiro': True},
        data_fits_in_memory=False,
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
        batch_size=256,
        loss='mean_absolute_error',
        scoring=r2_score,
        metrics=['mae', 'mse'],
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_learning_rate=True,
        model_checkpoints={
            'save_best_only': False,
            'save_freq': 'epoch',
            'save_weights_only': False
        },
        save_val_pred_history=True
    )
    RN1_R090_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.9
        },
    )
    RN1_R080_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.8
        },
    )
    RN1_R070_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.7
        },
    )
    RN1_R060_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.6
        },
    )
    RN1_R050_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.5
        },
    )
    RN1_R040_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.4
        },
    )
    RN1_R030_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.3
        },
    )
    RN1_R020_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.2
        },
    )
    RN1_R010_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.1
        },
    )
    RN1_R008_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.08
        },
    )
    RN1_R006_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.06
        },
    )
    RN1_R004_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.04
        },
    )
    RN1_R002_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.02
        },
    )

    RN1_R100_AGE_SEX = Experiment(
        description='Predict age and sex using the 4-block ResNet.',
        model=resnet_v1,
        model_kwargs={},
        extractor=SourceTask,
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 1.0
        },
        extractor_labels={'age': True, 'sex': True},
        extractor_features={'mode': 'raw', 'ribeiro': True},
        data_fits_in_memory=False,
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
        batch_size=256,
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
        model_checkpoints={
            'save_best_only': False,
            'save_freq': 'epoch',
            'save_weights_only': False
        },
        save_val_pred_history=True
    )
    RN1_R100_AGE_SCALED = RN1_R100_AGE._replace(
        description='Scales the age target by a factor 100.',
        extractor_labels={'age': True, 'sex': False, 'scale_age': True},
    )

    RN2_R100_SEX = RN1_R100_SEX._replace(
        description='A 12 block ResNet architecture from Gustafsson et al. '
                    'Trained here to predict sex using the raw ECG signal.',
        model=resnet_v2,
    )

    RN2_R100_AGE = RN1_R100_AGE._replace(
        description='Using the RN2 architecture to predict age.',
        model=resnet_v2
    )
    RN2_R090_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.9
        },
    )
    RN2_R080_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.8
        },
    )
    RN2_R070_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.7
        },
    )
    RN2_R060_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.6
        },
    )
    RN2_R050_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.5
        },
    )
    RN2_R040_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.4
        },
    )
    RN2_R030_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.3
        },
    )
    RN2_R020_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.2
        },
    )
    RN2_R010_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.1
        },
    )
    RN2_R008_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.08
        },
    )
    RN2_R006_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.06
        },
    )
    RN2_R004_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.04
        },
    )
    RN2_R002_AGE = RN1_R100_AGE._replace(
        extractor_index={
            'exclude_train_aliases': True,
            'train_percent': 0.02
        },
    )

    RN2_R100_AGE_SEX = RN1_R100_AGE_SEX._replace(
        description='Using the RN2 architecture to predict age and sex.',
        model=resnet_v2
    )


class PTBXL(Experiment, Enum):
    # {model}_{sample_rate}_{targets}
    XRN50_V1_100HZ_ALL = Experiment(
        description='Using the xresnet50 architecture to predict all 71 '
                    'labels in PTBXL',
        model=xrn50,
        model_kwargs={
            'initial_bn': True,
            'augmentation': {'mode': 'sample'},
        },
        extractor=PTBXLExtractor,
        extractor_features={'leads': 8, 'resolution': 'low'},
        data_fits_in_memory=True,
        optimizer=torch.optim.AdamW,
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-4,
                'warmup_target': 1e-3,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 90
            }
        },
        epochs=100,
        batch_size=256,
        loss=torch.nn.BCEWithLogitsLoss,
        scoring=roc_auc_score,
        metrics=['auc'],
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        use_tensorboard=True,
        save_model=True,
        model_checkpoints=True,
        save_learning_rate=True,
        save_val_pred_history=True
    )
    XRN50_V2_100HZ_ALL = XRN50_V1_100HZ_ALL._replace(
        description='Change augmentation to batch-wise sliding windows.',
        model_kwargs={
            'initial_bn': True,
            'augmentation': {'mode': 'batch'},
        }
    )
    XRN50_V3_100HZ_ALL = XRN50_V1_100HZ_ALL._replace(
        description='Turn off augmentation.',
        model_kwargs={
            'initial_bn': True,
        }
    )
    XRN50_V4_100HZ_ALL = XRN50_V1_100HZ_ALL._replace(
        description='Turn off initial batch-norm layer.',
        model_kwargs={
            'initial_bn': False,
            'augmentation': {'mode': 'sample'}
        }
    )
    XRN50_V5_100HZ_ALL = XRN50_V1_100HZ_ALL._replace(
        description='Change from AdamW to Adam optimizer.',
        optimizer=torch.optim.Adam
    )
    XRN50_V1_500HZ_ALL = XRN50_V1_100HZ_ALL._replace(
        description='Use the high-resolution ECGs instead (this automatically '
                    'adjusts the initial conv-layer settings)',
        extractor_features={'leads': 8, 'resolution': 'high'},
    )

    PTA100_XRN50V1_500HZ_ALL = Experiment(
        description='XRN50 pretrained on age, finetuning on PTBXL.',
        model=pretrained_pt,
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'XRN50V1_R100_AGE',
                'commit': '',
                'epoch': 26,
                'trainable': False,
            }
        },
        extractor=PTBXLExtractor,
        extractor_features={'leads': 8, 'resolution': 'high'},
        data_fits_in_memory=True,
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
        building_model_requires_development_data=True,
        use_predefined_splits=True,
        loss=torch.nn.BCEWithLogitsLoss,
        scoring=roc_auc_score,
        metrics=['auc'],
        use_tensorboard=True,
        save_learning_rate=True,
        save_val_pred_history=True
    )
    PTA100_XRN50V2_500HZ_ALL = PTA100_XRN50V1_500HZ_ALL._replace(
        description='V2.',
        model_kwargs={
            'from_xp': {
                'xp_project': 'transfer',
                'xp_base': 'Source',
                'xp_name': 'XRN50V2_R100_AGE',
                'commit': '',
                'epoch': 47,
                'trainable': False,
            }
        },
    )
    PTA100_XRN50V3_500HZ_ALL = PTA100_XRN50V1_500HZ_ALL._replace(
        description='Increased lr',
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-3,
                'warmup_target': 1e-2,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 30
            }
        },
    )
    PTA100_XRN50V4_500HZ_ALL = PTA100_XRN50V2_500HZ_ALL._replace(
        description='Augmentation + increased lr',
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 1e-3,
                'warmup_target': 1e-2,
                'alpha': 0.01,
                'warmup_epochs': 10,
                'decay_epochs': 30
            }
        },
    )
