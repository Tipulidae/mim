from enum import Enum

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import roc_auc_score

from mim.experiments.experiments import Experiment
from mim.extractors.esc_trop import EscTrop
from mim.models.simple_nn import ecg_cnn, serial_ecg, ffnn
from mim.models.load import pre_process_using_xp
from mim.cross_validation import ChronologicalSplit


# Here's an attempt at a structure for experiment names:
# [Data source]_[[features]]_[target]_[model]_[version]
# Data source: {ESC, EXPECT, PTB, PTBXL, ...}
# Features: {B, R}#, where B=beat, R=raw and # is the number of records used.
# Other common feature-sets to be given names as needed.
# Target: {MACE,AMI,...}#, where # would indicate some time frame
# Model: Some short hand for the models that I use
# Version: If I run multiple variations of this experiment, a version is handy

# {R/B}{#}_{BASELINE, }


class ESCT(Experiment, Enum):
    # Serial ECG analysis experiments on the ESC-Trop data
    B1_CNN1 = Experiment(
        description='Baseline CNN model using only current ECG median beat to '
                    'predict MACE within 30 days.',
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
                'dense': True,
                'dense_size': 10,
                'downsample': False
            }
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
        cv_kwargs={'test_size': 1/3},
        scoring=roc_auc_score,
    )

    R1_CNN1 = B1_CNN1._replace(
        description='Basic CNN model using only current ECG raw signal to '
                    'predict MACE within 30 days.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0']
            },
        },
    )

    R1_NOTCH_CNN1 = B1_CNN1._replace(
        description='Uses notch-filter and clipping to remove outliers and '
                    'baseline wander. Slightly increases dropout to '
                    'compensate for less regularization overall.',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'dropout': 0.5,
                'filter_first': 32,
                'filter_last': 32,
                'kernel_first': 16,
                'kernel_last': 16,
                'pool_size': 16,
                'batch_norm': True,
                'dense': True,
                'dense_size': 10
            }
        },
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0']
            },
            "processing": {
                'notch-filter',
                'clip_outliers'
            },
        }
    )

    B1_CNN1_SANITY1 = B1_CNN1._replace(
        description='Try to predict mace 30 using only the old ecg...',
        extractor_kwargs={
            'features': {
                'ecg_mode': 'beat',
                'ecgs': ['ecg_1']
            },
        },
    )

    B1_AGE_SEX_CNN1 = B1_CNN1._replace(
        description='Baseline CNN model using a 2 conv layer network on '
                    '1 ECG median beat plus age and sex features concatenated '
                    'at the end, predicting MACE within 30 days.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'beat',
                'ecgs': ['ecg_0'],
                'flat_features': ['age', 'male']
            }
        },
    )

    B2_CNN1 = B1_CNN1._replace(
        description='Running two CNNs in parallel on two ECG beat signals. ',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'beat',
                'ecgs': ['ecg_0', 'ecg_1']
            },
        },
    )

    R1_CNN2 = R1_CNN1._replace(
        description='Best model after hyperband tuning. ',
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'dropout': 0.4,
                'filter_first': 44,
                'filter_last': 31,
                'kernel_first': 5,
                'kernel_last': 7,
                'batch_norm': False,
                'dense': True,
                'downsample': True,
                'dense_size': 10
            }
        },
        epochs=200,
        batch_size=128,
    )

    R1_CNN2a = R1_CNN1._replace(
        description='Playing around with dense layers. Also learning rate '
                    'decay.',
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'dropout': 0.4,
                'filter_first': 44,
                'filter_last': 31,
                'kernel_first': 5,
                'kernel_last': 7,
                'batch_norm': False,
                'dense': True,
                'downsample': True,
                'dense_size': 100,
            },
            'cat_dense_size': 20,
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                # 'learning_rate': 1e-4
                'learning_rate': {
                    'scheduler': ExponentialDecay,
                    'scheduler_kwargs': {
                        'initial_learning_rate': 3e-4,
                        'decay_steps': 76,  # Every epoch
                        'decay_rate': 0.99  # Decay by 1%
                    }
                },
            }
        },
        epochs=200,
        batch_size=128,
    )

    R1_CNN2b = R1_CNN1._replace(
        description='Best model after hyperband tuning, but with batch-norm.',
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'dropout': 0.4,
                'filter_first': 44,
                'filter_last': 31,
                'kernel_first': 5,
                'kernel_last': 7,
                'batch_norm': True,
                'dense': True,
                'downsample': True,
                'dense_size': 10
            }
        },
        epochs=500,
        batch_size=128,
    )

    R2_CNN2 = R1_CNN2._replace(
        description='Trying the same model but with two ECGs.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1']
            },
        },
    )

    R1_NOTCH_CNN2 = R1_CNN2._replace(
        description='Trying the best model with pre-processing.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0']
            },
            "processing": {
                'notch-filter',
                'clip_outliers'
            },
        }
    )

    R2_ECNN2 = Experiment(
        description='Loads the pre-trained R1_CNN2 model and uses it '
                    'to encode each input ECG. The resulting vectors '
                    'are stacked (concatenated) and fed to a FFNN with '
                    'two hidden layers. The pre-trained model is not updated.',
        model=ffnn,
        model_kwargs={
            'dense_layers': [20, 10],
            'dropout': 0.3
        },
        pre_processor=pre_process_using_xp,
        pre_processor_kwargs={
            'xp_name': 'ESCT/R1_CNN2',
            'commit': 'bb076fbf682a9df358ed9a2c6731ad7fc108d1f5',
            'final_layer_index': -2
        },
        epochs=300,
        batch_size=64,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-4}
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1']
            },
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1/3},
        scoring=roc_auc_score,
    )

    R2_ECNN2_DT = R2_ECNN2._replace(
        description='Uses 2 raw ECG signals with the pre-trained CNN, and '
                    'adds time since last ecg (log_dt) as feature in the '
                    'end. ',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'features': ['log_dt']
            },
        },
    )

    R2_ECNN2_AGE_SEX = R2_ECNN2._replace(
        description='Two ECGs + age + sex, but no delta-t.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'features': ['age', 'male']
            },
        },
    )

    R2_ECNN2_DT_AGE_SEX = R2_ECNN2._replace(
        description='Adds age and sex as features in addition to delta-t.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'features': ['age', 'male', 'log_dt']
            },
        },
    )

    R1_CNN2_DT = R1_CNN2._replace(
        description='Wonder what happens if we add time since last ECG but '
                    'without the old ECG...',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'features': ['log_dt']
            },
        },
    )

    R2_D100 = Experiment(
        description='',
        model=serial_ecg,
        model_kwargs={
            'feature_extraction': {
                'xp_name': 'MultipleECG/R1_TUNED_D100',
                'commit': 'b515785522118ad7f6b78df95fdd102904f29b8f',
                'final_layer_index': -2,
            },
            'dense_size': 20
        },
        epochs=300,
        batch_size=64,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-4}
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['index', 'old']
            },
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1/3},
        scoring=roc_auc_score,
    )
