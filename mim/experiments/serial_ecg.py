from enum import Enum

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import roc_auc_score

from mim.experiments.experiments import Experiment
from mim.extractors.esc_trop import EscTrop
from mim.models.simple_nn import basic_cnn, serial_ecg
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
                'dense_size': 10
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
                'dense_size': 10
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
            'dense_size': 10
        },
        epochs=500,
        batch_size=128,
    )

    R1_TUNED_D100 = BASELINE_RAW._replace(
        description='Playing around with dense layers. Also learning rate '
                    'decay.',
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
            'dense_size': 100,
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
            'dense_size': 10
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

    R2_PRETRAINED = Experiment(
        description='Loads the pre-trained R1_TUNED model and uses it for '
                    'feature engineering on each input ECG. The resulting '
                    'vectors are stacked (concatenated) and fed to FFNN with '
                    'one hidden layer. The pre-trained model is not updated.',
        model=serial_ecg,
        model_kwargs={
            'feature_extraction': {
                'xp_name': 'MultipleECG/R1_TUNED',
                'commit': 'f7dffd5ed9f3b98e6b1666d238a3933f551cf9fe'
            }
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

    R2_PT_DT = R2_PRETRAINED._replace(
        description='Uses 2 raw ECG signals with the pre-trained CNN, and '
                    'adds time since last ecg (delta-t) as feature in the '
                    'end. ',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['index', 'old'],
                'features': ['delta_t']
            },
        },
    )

    R2_PT_AGE_SEX = R2_PRETRAINED._replace(
        description='Two ECGs + age + sex, but no delta-t.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['index', 'old'],
                'features': ['age', 'sex']
            },
        },
    )

    R2_PT_DT_AGE_SEX = R2_PT_DT._replace(
        description='Adds age and sex as features in addition to delta-t.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['index', 'old'],
                'features': ['age', 'sex', 'delta_t']
            },
        },
    )

    R2_PRETRAINED_DIFF = R2_PRETRAINED._replace(
        description='Uses R1_TUNED as feature engineering, then combines the '
                    'outputs from each ECG by stacking the differences from '
                    'the first ECG together. ',
        model_kwargs={
            'feature_extraction': {
                'xp_name': 'MultipleECG/R1_TUNED',
                'commit': 'f7dffd5ed9f3b98e6b1666d238a3933f551cf9fe'
            },
            'combiner': 'diff'
        },
    )

    R2_PRETRAINED_DIFF2 = R2_PRETRAINED_DIFF._replace(
        description='This time, allow backpropagation to update the '
                    'pre-trained feature-extraction model.',
        model_kwargs={
            'feature_extraction': {
                'xp_name': 'MultipleECG/R1_TUNED',
                'commit': 'f7dffd5ed9f3b98e6b1666d238a3933f551cf9fe',
                'trainable': True
            },
            'combiner': 'diff'
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': ExponentialDecay,
                    'scheduler_kwargs': {
                        'initial_learning_rate': 1e-4,
                        'decay_steps': 152,  # Corresponds to once per epoch
                        'decay_rate': 0.96
                    }
                },
            }
        },
        epochs=10
    )

    R1_TUNED_DT = R1_TUNED._replace(
        description='Wonder what happens if we add time since last ECG but '
                    'without the old ECG...',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['index'],
                'features': ['delta_t']
            },
        },
    )

    R1_TUNED_DT2 = R1_TUNED_DT._replace(
        description='Is there a difference if we use the pre-trained network '
                    'instead?',
        model=serial_ecg,
        model_kwargs={
            'feature_extraction': {
                'xp_name': 'MultipleECG/R1_TUNED',
                'commit': 'f7dffd5ed9f3b98e6b1666d238a3933f551cf9fe'
            },
            'number_of_ecgs': 1
        },
        epochs=300
    )

    R1_TUNED_DT_AGE_SEX = R1_TUNED_DT2._replace(
        description='Same as R2_PT_DT_AGE_SEX, but uses only the first ECG.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['index'],
                'features': ['age', 'sex', 'delta_t']
            },
        },
    )

    R1_TUNED_AGE_SEX = R1_TUNED_DT2._replace(
        description='Index ECG + age + sex.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['index'],
                'features': ['age', 'sex']
            },
        },
    )

    FOO = R1_TUNED_DT2._replace(
        description='Index ECG + age + sex.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['index'],
                'features': ['age', 'sex']
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

# b515785522118ad7f6b78df95fdd102904f29b8f
