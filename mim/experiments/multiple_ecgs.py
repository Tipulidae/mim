from enum import Enum

from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

from mim.experiments.experiments import Experiment
from mim.extractors.esc_trop import EscTrop
from mim.models.simple_nn import basic_cnn
from mim.cross_validation import ChronologicalSplit


# Here's an attempt at a structure for experiment names:
# [Data source]_[[features]]_[target]_[model]_[version]
# Data source: {ESC, EXPECT, PTB, PTBXL, ...}
# Features: {B, R}#, where B=beat, R=raw and # is the number of records used.
# Other common feature-sets to be given names as needed.
# Target: {MACE,AMI,...}#, where # would indicate some time frame
# Model: Some short hand for the models that I use
# Version: If I run multiple variations of this experiment, a version is handy


class MultipleECG(Experiment, Enum):
    BASELINE_BEAT = Experiment(
        description='Baseline CNN model using only current ECG median beat to '
                    'predict MACE within 30 days.',
        model=basic_cnn,
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
                'output_size': 10
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
                'ecgs': ['index']
            },
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1/3},
        scoring=roc_auc_score,
    )

    BASELINE_RAW = BASELINE_BEAT._replace(
        description='Basic CNN model using only current ECG raw signal to '
                    'predict MACE within 30 days.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['index']
            },
        },
    )

    BASELINE_NOTCH = BASELINE_BEAT._replace(
        description='Uses notch-filter and clipping to remove outliers and '
                    'baseline wander. Slightly increases dropout to '
                    'compensate for less regularization overall.',
        model=basic_cnn,
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
                'output_size': 10
            }
        },
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['index']
            },
            "processing": {
                'notch-filter',
                'clip_outliers'
            },
        }
    )

    SANITY1 = BASELINE_BEAT._replace(
        description='Try to predict mace 30 using only the old ecg...',
        extractor_kwargs={
            'features': {
                'ecg_mode': 'beat',
                'ecgs': ['old']
            },
        },
    )

    BASELINE_AGE_SEX = BASELINE_BEAT._replace(
        description='Baseline CNN model using a 2 conv layer network on '
                    '1 ECG median beat plus age and sex features concatenated '
                    'at the end, predicting MACE within 30 days.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'beat',
                'ecgs': ['index'],
                'features': ['age', 'sex']
            }
        },
    )

    BASELINE_DOUBLE_ECG = BASELINE_BEAT._replace(
        description='Running two CNNs in parallel on two ECG beat signals. ',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'beat',
                'ecgs': ['index', 'old']
            },
        },
    )

    R1_TUNED = BASELINE_RAW._replace(
        description='Best model after hyperband tuning. ',
        model_kwargs={
            'num_layers': 2,
            'dropout': 0.4,
            'filter_first': 44,
            'filter_last': 31,
            'kernel_first': 5,
            'kernel_last': 7,
            'batch_norm': False,
            'dense': True,
            'downsample': True,
            'output_size': 10
        },
        epochs=500,
        batch_size=128,
    )

    R1_TUNED_BN = R1_TUNED._replace(
        description='Best model after hyperband tuning, but with batch-norm.',
        model_kwargs={
            'num_layers': 2,
            'dropout': 0.4,
            'filter_first': 44,
            'filter_last': 31,
            'kernel_first': 5,
            'kernel_last': 7,
            'batch_norm': True,
            'dense': True,
            'downsample': True,
            'output_size': 10
        },
        epochs=500,
        batch_size=128,
    )

    R2_TUNED = R1_TUNED._replace(
        description='Trying the same model but with two ECGs.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['index', 'old']
            },
        },
    )

    R1_TUNED_NOTCH = R1_TUNED._replace(
        description='Trying the best model with pre-processing.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['index']
            },
            "processing": {
                'notch-filter',
                'clip_outliers'
            },
        }
    )
