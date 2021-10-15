# -*- coding: utf-8 -*-

from enum import Enum

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from mim.experiments.experiments import Experiment
from mim.extractors.esc_trop import EscTrop
from mim.extractors.extractor import sklearn_process
from mim.models.simple_nn import (
    ecg_cnn,
    ffnn,
    logistic_regression,
    logistic_regression_ab,
    pretrained_resnet
)
from mim.models.load import (
    pre_process_using_xp,
    load_ribeiro_model,
    load_model_from_experiment_result
)
from mim.cross_validation import ChronologicalSplit


class ESCT(Experiment, Enum):
    # RANDOM FOREST, FLAT FEATURES:
    M_RF1_DT = Experiment(
        description='Predicting MACE with Random Forest, using only the '
                    'time since last ECG.',
        model=RandomForestClassifier,
        model_kwargs={
            'n_estimators': 1000,
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt']
            },
        },
        building_model_requires_development_data=False,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_RF1_AGE = M_RF1_DT._replace(
        description='Predicting MACE with Random Forest, using only the '
                    'patient age.',
        extractor_kwargs={
            "features": {
                'flat_features': ['age']
            },
        },
    )
    M_RF1_SEX = M_RF1_DT._replace(
        description='Predicting MACE with Random Forest, using only the '
                    'patient sex.',
        extractor_kwargs={
            "features": {
                'flat_features': ['male']
            },
        },
    )
    M_RF1_TNT = M_RF1_DT._replace(
        description='Predicting MACE with Random Forest, using only the '
                    'first TnT lab measurement.',
        extractor_kwargs={
            "features": {
                'flat_features': ['tnt_1']
            },
        },
    )
    M_RF1_DT_AGE = M_RF1_DT._replace(
        description='Predicting MACE with Random Forest, using only the '
                    'time since last ECG and patient age.',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age']
            },
        },
    )
    M_RF1_DT_AGE_SEX = M_RF1_DT._replace(
        description='Predicting MACE with Random Forest, using only the '
                    'time since last ECG, patient age and sex.',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male']
            },
        },
    )
    M_RF1_DT_AGE_SEX_TNT = M_RF1_DT._replace(
        description='Predicting MACE with Random Forest, using only the '
                    'time since last ECG, age, sex and first TnT measurement.',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
        },
    )

    # AB Log Reg, Flat Features:
    AB_M_LR1_DT_AGE_SEX_LOGTNT = Experiment(
        description="foo",
        model=logistic_regression_ab,
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'flat_features': ['log_tnt_1', "age", "male", "log_dt"]
            },
        },
        epochs=300,
        batch_size=-1,
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1},
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )

    # KERAS LOGISTIC REGRESSION, FLAT FEATURES:
    M_LR1_DT = Experiment(
        description='Logistic regression, mace vs dt',
        model=logistic_regression,
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt']
            },
        },
        epochs=300,
        batch_size=-1,
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1},
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_LR1_AGE = M_LR1_DT._replace(
        description='Logistic regression, mace vs dt',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt']
            },
        },
    )
    M_LR1_SEX = M_LR1_DT._replace(
        description='Logistic regression, mace vs sex',
        extractor_kwargs={
            "features": {
                'flat_features': ['male']
            },
        },
    )
    M_LR1_TNT = M_LR1_DT._replace(
        description='Logistic regression, mace vs tnt',
        extractor_kwargs={
            "features": {
                'flat_features': ['tnt_1']
            },
        },
    )
    M_LR1_LOGTNT = M_LR1_DT._replace(
        description='Logistic regression, mace vs log-tnt',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_tnt_1']
            },
        },
    )
    M_LR1_DT_AGE = M_LR1_DT._replace(
        description='Logistic regression, mace vs dt + age',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age']
            },
        },
    )
    M_LR1_DT_AGE_SEX = M_LR1_DT._replace(
        description='Logistic regression, mace vs dt + age + sex',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male']
            },
        },
    )
    M_LR1_DT_AGE_SEX_TNT = M_LR1_DT._replace(
        description='Logistic regression, mace vs dt + age + sex + tnt',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
        },
    )
    M_LR1_FF = M_LR1_DT._replace(
        description='Logistic regression, mace vs all the flat-features',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_tnt_1', 'male', 'age', 'log_dt']
            },
        },
    )

    # SKLEARN LOGISTIC REGRESSION, FLAT FEATURES:
    M_LR2_DT_AGE_SEX_TNT = Experiment(
        description='Scikit-learns logistic regression model, mace vs flat '
                    'features.',
        model=LogisticRegression,
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
        },
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )

    M_F1_LR2 = Experiment(
        description='Scikit-learn logistic regression model, mace vs '
                    'features from Forberg et al. Features are normalized '
                    'and then reduced in dimension by PCA. ',
        model=LogisticRegression,
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'forberg': ['ecg_0']
            },
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'forberg_features': {
                'ecg_0': {
                    'processor': 'Pipeline',
                    'steps': [
                        ('scaler', StandardScaler, {}),
                        ('pca', PCA, {
                            'n_components': 100,
                            'whiten': False,
                            'random_state': 42
                        })
                    ]
                }
            }
        },
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_F2_LR2 = M_F1_LR2._replace(
        description='Scikit-learn logistic regression model, mace vs '
                    'features from Forberg et al. Features are normalized '
                    'and then reduced in dimension by PCA. ',
        extractor_kwargs={
            "features": {
                'forberg': ['ecg_0', 'diff']
            },
        },
        pre_processor_kwargs={
            'forberg_features': {
                'ecg_0': {
                    'processor': 'Pipeline',
                    'steps': [
                        ('scaler', StandardScaler, {}),
                        ('pca', PCA, {
                            'n_components': 145,
                            'whiten': False,
                            'random_state': 42
                        })
                    ]
                },
                'diff': {
                    'processor': 'Pipeline',
                    'steps': [
                        ('scaler', StandardScaler, {}),
                        ('pca', PCA, {
                            'n_components': 145,
                            'whiten': False,
                            'random_state': 42
                        })
                    ]
                },
            },
        },
    )
    M_F1_LR2_FF = M_F1_LR2._replace(
        description='Scikit-learn logistic regression model, mace vs '
                    'features from Forberg et al. Features are normalized '
                    'and then reduced in dimension by PCA.',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male', 'log_tnt_1'],
                'forberg': ['ecg_0']
            },
        },
        pre_processor_kwargs={
            'forberg_features': {
                'ecg_0': {
                    'processor': 'Pipeline',
                    'steps': [
                        ('scaler', StandardScaler, {}),
                        ('pca', PCA, {
                            'n_components': 2,
                            'whiten': False,
                            'random_state': 42
                        })
                    ]
                }
            },
            'flat_features': {
                'processor': StandardScaler,
            }
        },
    )
    M_F2_LR2_FF = M_F1_LR2._replace(
        description='Scikit-learn logistic regression model, mace vs '
                    'features from Forberg et al. Features are normalized '
                    'and then reduced in dimension by PCA. I tried a bunch '
                    'of settings for dimension, ~145 gave the best AUC.',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male', 'log_tnt_1'],
                'forberg': ['ecg_0', 'diff']
            },
        },
        pre_processor_kwargs={
            'forberg_features': {
                'ecg_0': {
                    'processor': 'Pipeline',
                    'steps': [
                        ('scaler', StandardScaler, {}),
                        ('pca', PCA, {
                            'n_components': 5,
                            'whiten': False,
                            'random_state': 42
                        })
                    ]
                },
                'diff': {
                    'processor': 'Pipeline',
                    'steps': [
                        ('scaler', StandardScaler, {}),
                        ('pca', PCA, {
                            'n_components': 5,
                            'whiten': False,
                            'random_state': 42
                        })
                    ]
                }
            },
            'flat_features': {
                'processor': StandardScaler,
            }
        },
    )

    # SINGLE RAW ECG, CNN VARIATIONS:
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
                'downsample': False
            },
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.3],
                'batch_norms': [False]
            },
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
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0']
            },
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R1_CNN2 = M_R1_CNN1._replace(
        description='Try adjusting the final dense-layer size from 10 to 100.'
                    'Also downsamples the ECG first, to its original 500Hz.',
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
                'down_sample': True
            },
            'final_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.3],
                'batch_norms': [False]
            },
        }
    )
    M_R1_CNN3 = M_R1_CNN2._replace(
        description='Add class-weights to the training, and also reduce '
                    'learning-rate when validation loss plateaus.',
        class_weight={0: 1, 1: 10},
        reduce_lr_on_plateau={
            'monitor': 'val_loss',
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-6
        },
    )
    M_R1_NOTCH_CNN3 = M_R1_CNN3._replace(
        description='Uses notch-filter and clipping to remove outliers and '
                    'baseline wander.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0']
            },
            "processing": {
                'notch-filter',
                'clip_outliers'
            }
        },
    )
    M_R1_CNN4 = M_R1_CNN2._replace(
        description='Increasing dropout, trying out a new lr schedule',
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
                'down_sample': True
            },
            'final_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.5],
                'batch_norms': [False]
            },
        },
        class_weight={0: 1, 1: 10},
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [153*20, 153*40, 153*150],
                        'values': [1e-3, 1e-4, 1e-5, 1e-6],
                    }
                },
            }
        },
    )
    M_R1_CNN5 = M_R1_CNN4._replace(
        description='Adjusting pool-size and kernel-size.',
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'dropout': 0.5,
                'filter_first': 32,
                'filter_last': 32,
                'kernel_first': 32,
                'kernel_last': 16,
                'pool_size': 16,
                'batch_norm': True,
                'down_sample': True
            },
            'final_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.5],
                'batch_norms': [False]
            },
        },
    )
    M_R1_CNN6 = M_R1_CNN4._replace(
        description='Increasing dropout even further',
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'dropout': 0.7,
                'filter_first': 32,
                'filter_last': 32,
                'kernel_first': 16,
                'kernel_last': 16,
                'pool_size': 16,
                'batch_norm': True,
                'down_sample': True
            },
            'final_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.5],
                'batch_norms': [False]
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [153 * 50, 153 * 100],
                        'values': [1e-3, 1e-4, 1e-5],
                    }
                },
            }
        },
    )

    # CNN4, 1 ECG + FLAT FEATURES
    M_R1_CNN4_DT = M_R1_CNN4._replace(
        description='Adds the (logarithm of the) time since last ECG.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt']
            },
        },
    )
    M_R1_CNN4_AGE = M_R1_CNN4._replace(
        description='Adds age.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['age']
            },
        },
    )
    M_R1_CNN4_SEX = M_R1_CNN4._replace(
        description='Adds sex (1 = male, 0 = female).',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['male']
            },
        },
    )
    M_R1_CNN4_TNT = M_R1_CNN4._replace(
        description='Adds the first TnT lab measurement.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['tnt_1']
            },
        },
    )
    M_R1_CNN4b_TNT = M_R1_CNN4_TNT._replace(
        description='Adjusts the learning-rate schedule. Also, what if we '
                    'skip the class-weights?',
        epochs=100,
        class_weight=None,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [153 * 5],
                        'values': [1e-3, 1e-4],
                    }
                },
            }
        },
    )
    M_R1_CNN4_DT_AGE = M_R1_CNN4._replace(
        description='Adds time since last ECG and age.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age']
            },
        },
    )
    M_R1_CNN4_DT_AGE_SEX = M_R1_CNN4._replace(
        description='Adds time since last ECG, age and sex.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male']
            },
        },
    )
    M_R1_CNN4_DT_AGE_SEX_TNT = M_R1_CNN4._replace(
        description='Adds time since last ECG, age, sex and TnT.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
        },
    )

    # CNN7 variations, best random-search model for 1ECG+FF
    M_R1_CNN7 = Experiment(
        description='Uses xp_379 from M_R1_FF_CNN_RS, which was the top '
                    'performing model found after 400 iterations of random '
                    'search. In the random search, flat-features were '
                    'included, but in this experiment we only use the ECG.',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': True,
                'num_layers': 3,
                'dropouts': [0.3, 0.5, 0.2],
                'pool_size': 8,
                'filter_first': 12,
                'filter_last': 16,
                'kernel_first': 13,
                'kernel_last': 41,
                'batch_norms': [True, True, False],
                'weight_decays': [0.01, 0.001, 0.01],
            },
            'ecg_ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.2],
                'batch_norms': [False]
            },
            'ecg_comb_ffnn_kwargs': None,
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [20],
                'dropouts': [0.2],
                'batch_norms': [False]
            }
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.001}
        },
        epochs=200,
        batch_size=64,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1/3},
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    M_R1_CNN7_FF = M_R1_CNN7._replace(
        description='Uses the CNN7 (or xp_379) model with flat-features, as '
                    'intended. This should replicate the results from '
                    'experiment xp_379.',
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
        },
    )
    M_R2_CNN7_FF = M_R1_CNN7._replace(
        description='Uses the CNN7 (or xp_379) model, except with 2 ECGs '
                    'instead of just one. The CNN combiner defaults to '
                    'concatenation, but there is no additional ecg_ffnn '
                    'added, which might be a useful variation.',
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
        },
    )
    MC_R1_CNN7_FF = M_R1_CNN7_FF._replace(
        description='Like M_R1_CNN7_FF, except the target is now the '
                    'components of MACE, divided into chapters. So a multi-'
                    'label problem.',
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
            'labels': {
                'target': 'mace_chapters'
            }
        },
        epochs=100,
    )

    # CNN7 variations with different random seeds
    M_R1_CNN7_FF_r1 = M_R1_CNN7_FF._replace(
        description='Same as M_R1_CNN7_FF, but with a different random seed.',
        random_state=1000,
    )
    M_R1_CNN7_FF_r2 = M_R1_CNN7_FF._replace(
        description='Same as M_R1_CNN7_FF, but with a different random seed.',
        random_state=1001,
    )
    M_R1_CNN7_FF_r3 = M_R1_CNN7_FF._replace(
        description='Same as M_R1_CNN7_FF, but with a different random seed.',
        random_state=1002,
    )
    M_R1_CNN7_FF_r4 = M_R1_CNN7_FF._replace(
        description='Same as M_R1_CNN7_FF, but with a different random seed.',
        random_state=1003,
    )
    M_R1_CNN7_FF_r5 = M_R1_CNN7_FF._replace(
        description='Same as M_R1_CNN7_FF, but with a different random seed.',
        random_state=1004,
    )
    M_R1_CNN7_FF_r6 = M_R1_CNN7_FF._replace(
        description='Same as M_R1_CNN7_FF, but with a different random seed.',
        random_state=1005,
    )
    M_R1_CNN7_FF_r7 = M_R1_CNN7_FF._replace(
        description='Same as M_R1_CNN7_FF, but with a different random seed.',
        random_state=1006,
    )
    M_R1_CNN7_FF_r8 = M_R1_CNN7_FF._replace(
        description='Same as M_R1_CNN7_FF, but with a different random seed.',
        random_state=1007,
    )
    M_R1_CNN7_FF_r9 = M_R1_CNN7_FF._replace(
        description='Same as M_R1_CNN7_FF, but with a different random seed.',
        random_state=1008,
    )

    # CNN8 variations, best random-search model for 2ECG+FF
    M_R1_CNN8 = Experiment(
        description='Uses xp_294 from M_R2_FF_CNN_RS, which was the top '
                    'performing model found after 400 iterations of random '
                    'search. The model is optimized on 2 ECGs and flat-'
                    'features, but here I only use 1 ECG and no flat-'
                    'features.',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': True,
                'num_layers': 2,
                'dropouts': [0.3, 0.4],
                'pool_size': 22,
                'filter_first': 48,
                'filter_last': 48,
                'kernel_first': 9,
                'kernel_last': 21,
                'batch_norms': [False, True],
                'weight_decays': [0.1, 0.0],
            },
            'ecg_combiner': 'concatenate',
            'ecg_ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.2],
                'batch_norms': [False]
            },
            'ecg_comb_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.1],
                'batch_norms': [False]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.5],
                'batch_norms': [False]
            }
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.001}
        },
        epochs=200,
        batch_size=64,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1/3},
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    M_R1_CNN8_FF = M_R1_CNN8._replace(
        description='Uses the CNN8 model with 1 ECG and flat-features. ',
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
        }
    )
    M_R2_CNN8_FF = M_R1_CNN8._replace(
        description='Uses the CNN8 model with 2 ECGs and flat-features, '
                    'as it was optimized for. Should replicate xp_294 from '
                    'M_R2_FF_CNN_RS, with an AUC of ~0.8735.',
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
        }
    )

    # CNN8 variations with different random seeds
    M_R2_CNN8_FF_r1 = M_R2_CNN8_FF._replace(
        description='Same as M_R2_CNN8_FF, but with a different random seed.',
        random_state=1001,
    )
    M_R2_CNN8_FF_r2 = M_R2_CNN8_FF._replace(
        description='Same as M_R2_CNN8_FF, but with a different random seed.',
        random_state=1002,
    )
    M_R2_CNN8_FF_r3 = M_R2_CNN8_FF._replace(
        description='Same as M_R2_CNN8_FF, but with a different random seed.',
        random_state=1003,
    )
    M_R2_CNN8_FF_r4 = M_R2_CNN8_FF._replace(
        description='Same as M_R2_CNN8_FF, but with a different random seed.',
        random_state=1004,
    )
    M_R2_CNN8_FF_r5 = M_R2_CNN8_FF._replace(
        description='Same as M_R2_CNN8_FF, but with a different random seed.',
        random_state=1005,
    )
    M_R2_CNN8_FF_r6 = M_R2_CNN8_FF._replace(
        description='Same as M_R2_CNN8_FF, but with a different random seed.',
        random_state=1006,
    )
    M_R2_CNN8_FF_r7 = M_R2_CNN8_FF._replace(
        description='Same as M_R2_CNN8_FF, but with a different random seed.',
        random_state=1007,
    )
    M_R2_CNN8_FF_r8 = M_R2_CNN8_FF._replace(
        description='Same as M_R2_CNN8_FF, but with a different random seed.',
        random_state=1008,
    )
    M_R2_CNN8_FF_r9 = M_R2_CNN8_FF._replace(
        description='Same as M_R2_CNN8_FF, but with a different random seed.',
        random_state=1009,
    )

    # CNN9 variations, 2nd best random-search model for 1ECG+FF
    M_R1_CNN9 = Experiment(
        description='Uses xp_382 from M_R1_FF_CNN_RS, which was the second '
                    'best in terms of both AUC and rule-out. ',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'dropouts': [0.3, 0.4],
                'pool_size': 21,
                'filter_first': 52,
                'filter_last': 60,
                'kernel_first': 61,
                'kernel_last': 45,
                'batch_norms': [False, True],
                'weight_decays': [0.1, 0.0],
            },
            'ecg_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.2],
                'batch_norms': [False]
            },
            'ecg_comb_ffnn_kwargs': None,
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [20],
                'dropouts': [0.3],
                'batch_norms': [False]
            }
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.001}
        },
        epochs=100,
        batch_size=64,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1/3},
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    M_R1_CNN9_FF = M_R1_CNN9._replace(
        description='Uses the CNN9 model with 1 ECGs and flat-features, '
                    'as it was optimized for. Should replicate xp_382 from '
                    'M_R1_FF_CNN_RS, with an AUC of ~0.8726 and rule-out '
                    '~0.3122.',
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
        }
    )
    M_R2_CNN9_FF = M_R1_CNN9._replace(
        description='Uses the CNN9 model with 2 ECGs and flat-features.',
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
        }
    )

    # CNN10, best random-search model for 1 ECG
    M_R1_CNN10 = Experiment(
        description='Uses xp_210 from M_R1_CNN_RS, which was the second '
                    'best in terms of AUC, but looked better than the best '
                    'when considering the overall learning trend. ',
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
                'dropouts': [0.4],
                'batch_norms': [False]
            },
            'ecg_comb_ffnn_kwargs': None,
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.3],
                'batch_norms': [False]
            }
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.0001}
        },
        epochs=200,
        batch_size=64,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )

    # CNN11, best random-search model for 2 ECGs

    # ABs CNN MODEL, 1 ECG + FLAT FEATURES
    M_R1_AB1 = Experiment(
        description='Predicting MACE-30 using only single raw ECG, '
                    'using the CNN architecture from Anders '
                    'Björkelund et al. ',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': False,
                'num_layers': 3,
                'dropouts': [0.0, 0.3, 0.0],
                'kernels': [64, 16, 16],
                'filters': [64, 16, 8],
                'weight_decays': [1e-4, 1e-3, 1e-4],
                'pool_sizes': [32, 4, 8],
                'batch_norm': False,
            },
            'ecg_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.0],
                'batch_norms': [False]
            },
            'ecg_comb_ffnn_kwargs': None,
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.5],
                'batch_norms': [False]
            }
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 3e-3}
        },
        epochs=200,
        batch_size=64,
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R1_AB1_DT_AGE_SEX_TNT = M_R1_AB1._replace(
        description='Predicting MACE-30 using single raw ECG and flat-'
                    'features, using the CNN architecture from Anders '
                    'Björkelund et al. ',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
        },
    )
    M_R1_AB1_LOGTNT = M_R1_AB1._replace(
        description='Predicting MACE-30 using single raw ECG and log-tnt, '
                    'using the CNN architecture from Anders '
                    'Björkelund et al. ',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_tnt_1']
            },
        },
    )
    M_R1_AB1_FF = M_R1_AB1._replace(
        description='Predicting MACE-30 using single raw ECG and flat-'
                    'features, using the CNN architecture from Anders '
                    'Björkelund et al. ',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male', 'log_tnt_1']
            },
        },
    )

    # AB model but with 2 input CNNs
    M_R2_AB1 = M_R1_AB1._replace(
        description='Same AB model, but using 2 ECGs instead of one. Each '
                    'ECG trains its own separate network.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
            },
        },
    )
    M_R2_AB1_LOGTNT = M_R1_AB1._replace(
        description='Same AB model, 2 ECGs + log-tnt.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_tnt_1']
            },
        },
    )
    M_R2_AB1_FF = M_R1_AB1._replace(
        description='Same AB model, 2 ECGs + flat-features.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_tnt_1', 'male', 'age', 'log_dt']
            },
        },
    )

    # FFNN USING 2 ECGs PROCESSED WITH AB1
    M_R2_AB1_NN2 = Experiment(
        description='Loads the AB1-model trained on only a single ECG, and '
                    'use it as a pre-processor for two input ECGs. Here I '
                    'try a new ffnn-formula. ',
        model=ffnn,
        model_kwargs={
            'ecg_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.0],
                'batch_norms': [False]
            },
            'ecg_combiner': 'concatenate',
            'ecg_comb_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.3],
                'batch_norms': [False]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.5],
                'batch_norms': [False]
            }
        },
        pre_processor=pre_process_using_xp,
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_AB1',
            'commit': '1516289a38af2b63aaf3bd8352dff66a6b672a9d',
            'final_layer_index': -5  # This is the flatten-layer
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1']
            },
        },
        epochs=100,
        batch_size=64,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-4}
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R2_AB1_NN2_FF = M_R2_AB1_NN2._replace(
        description='Loads the AB1-model trained on only a single ECG, and '
                    'use it as a pre-processor for two input ECGs. Also use '
                    'flat-features.',
        model_kwargs={
            'ecg_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.0],
                'batch_norms': [False]
            },
            'ecg_combiner': 'difference',
            'ecg_comb_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.3],
                'batch_norms': [False]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.5],
                'batch_norms': [False]
            }
        },
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_tnt_1', 'male', 'age', 'log_dt']
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [153*10],
                        'values': [1e-3, 1e-4],
                    }
                },
            }
        },
    )
    M_R2_AB1b_NN2 = M_R2_AB1_NN2._replace(
        description='Loads the AB1-model trained on one ECG and flat-'
                    'features, and use the CNN-part to pre-process the '
                    'input ECGs.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
            },
        },
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_AB1_FF',
            'commit': '5557d9e6740724eeab7260b0f40068e47618a2d2',
            'final_layer_index': -8  # This is the flatten-layer
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-4}
        },
    )
    M_R2_AB1b_NN2_FF = M_R2_AB1_NN2._replace(
        description='Loads the AB1-model trained on one ECG and flat-'
                    'features, and use the CNN-part to pre-process the '
                    'input ECGs. Add flat-features too.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_tnt_1', 'male', 'age', 'log_dt']
            },
        },
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_AB1_FF',
            'commit': '5557d9e6740724eeab7260b0f40068e47618a2d2',
            'final_layer_index': -8,  # This is the flatten-layer,
            'input_key': 'ecg_0'
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-4}
        },
    )

    # LOGISTIC REGRESSION USING 1 ECG PROCESSED WITH CNN4 + FLAT FEATURES
    M_R1_CNN4_LR1_DT = Experiment(
        description='Pre-processing 1 input ECG with CNN4, into a single '
                    'scalar, then adding delta-t and feeding it into a '
                    'logistic-regression model.',
        model=logistic_regression,
        pre_processor=pre_process_using_xp,
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_CNN4',
            'commit': 'eb783dc3ab36f554b194cffe463919620d123496',
            'final_layer_index': -1
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt']
            },
        },
        epochs=300,
        batch_size=-1,
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1},
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R1_CNN4_LR1_AGE = M_R1_CNN4_LR1_DT._replace(
        description='ECG + Age',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['age']
            },
        },
    )
    M_R1_CNN4_LR1_SEX = M_R1_CNN4_LR1_DT._replace(
        description='ECG + Sex',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['male']
            },
        },
    )
    M_R1_CNN4_LR1_TNT = M_R1_CNN4_LR1_DT._replace(
        description='ECG + tnt',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['tnt_1']
            },
        },
    )
    M_R1_CNN4_LR1_DT_AGE = M_R1_CNN4_LR1_DT._replace(
        description='ECG + dt + age',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age']
            },
        },
    )
    M_R1_CNN4_LR1_DT_AGE_SEX = M_R1_CNN4_LR1_DT._replace(
        description='ECG + dt + age + sex',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male']
            },
        },
    )
    M_R1_CNN4_LR1_DT_AGE_SEX_TNT = M_R1_CNN4_LR1_DT._replace(
        description='ECG + age + sex + tnt',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
        },
    )

    # SINGLE RAW ECG, RESNET VARIATIONS
    M_R1_RN1 = Experiment(
        description="Pretrained ResNet architecture from Ribeiro et al.",
        model=pretrained_resnet,
        model_kwargs={
            'freeze_resnet': False,
            'ecg_ffnn_kwargs': None,
            'ecg_comb_ffnn_kwargs': None,
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': None
        },
        epochs=200,
        batch_size=32,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [305 * 20, 305 * 100],
                        'values': [1e-3, 1e-4, 1e-5],
                    }
                },
            }
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            }
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R1_RN2 = M_R1_RN1._replace(
        description="Pretrained ResNet, with class-weights",
        class_weight={0: 1, 1: 10.7},
    )
    M_R1_RN3 = M_R1_RN1._replace(
        description='Pretrained ResNet, but adding a final dense 100 at the '
                    'end.',
        model_kwargs={
            'freeze_resnet': False,
            'ecg_ffnn_kwargs': None,
            'ecg_comb_ffnn_kwargs': None,
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.0],
                'batch_norms': [False]
            }
        },
    )
    M_R1_RN4 = M_R1_RN1._replace(
        description='Pretrained ResNet, but adding final dense 100 layer with '
                    'dropout at the end.',
        model_kwargs={
            'freeze_resnet': False,
            'ecg_ffnn_kwargs': None,
            'ecg_comb_ffnn_kwargs': None,
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.3],
                'batch_norms': [False]
            }
        },
    )
    M_R1_RN5 = M_R1_RN1._replace(
        description='Adjusting the learning-rate scheduler and reducing epoch '
                    'number. ',
        model_kwargs={
            'freeze_resnet': False,
            'ecg_ffnn_kwargs': None,
            'ecg_comb_ffnn_kwargs': None,
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.0],
                'batch_norms': [False]
            }
        },
        epochs=50,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [305 * 10, 305 * 20],
                        'values': [1e-3, 1e-4, 1e-5],
                    }
                },
            }
        },
    )

    # SINGLE RAW ECG + FF, RESNET VARIATIONS

    # SINGLE RAW ECG + FF, RESNET 2-STEP MODEL
    M_R1_RN6a_FF = Experiment(
        description='Loads the pretrained Ribeiro ResNet model and use it as '
                    'a feature extractor for the input ECG. Feed this into a '
                    'dense 100 -> dense 10, then concatenate with ff and a '
                    'final dense 10 -> output. This is the first step of '
                    'the model, in which the resnet weights are frozen and '
                    'only the ffnn part is trained. ',
        model=pretrained_resnet,
        model_kwargs={
            'freeze_resnet': True,
            'ecg_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.5],
                'batch_norms': [False],
            },
            'ecg_comb_ffnn_kwargs': {
                'sizes': [20],
                'dropouts': [0.5],
                'batch_norms': [False]
            },
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.5],
                'batch_norms': [False],
            },
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            }
        },
        epochs=15,
        batch_size=32,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 1e-3
            },
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R1_RN6b_FF = Experiment(
        description='Loads the model from M_R1_RN6b_FF and unfreezes the '
                    'ResNet portion, fine-tuning the entire model.',
        model=load_model_from_experiment_result,
        model_kwargs={
            'xp_name': 'ESCT/M_R1_RN6a_FF',
            'commit': '9da74d5436820bbc29758ba07d5d1a83f60fc84f',
            'which': 'last',
            'trainable': True
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            }
        },
        epochs=100,
        batch_size=32,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 1e-4
            },
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )

    # FFNN USING 1 ECG PROCESSED WITH RESNET + FLAT FEATURES
    M_R1_RN5_NN1_DT_AGE_SEX_TNT = Experiment(
        description='Loads the pre-trained R1_RN5 model and uses it as a '
                    'feature extractor for the input ECG. Feed this into a '
                    'dense-100 layer, then concatenate some flat-features '
                    'before the final sigmoid layer.',
        model=ffnn,
        model_kwargs={
            'ecg_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.3],
                'batch_norms': [False]
            },
            'ecg_comb_ffnn_kwargs': None,
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': None
        },
        pre_processor=pre_process_using_xp,
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_RN5',
            'commit': '61fb8038d91ee119a1a889c3c86b27931f1f57b5',
            'which': 'last',
            'final_layer_index': -3
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            }
        },
        epochs=200,
        batch_size=32,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [305 * 100],
                        'values': [1e-3, 1e-4],
                    }
                },
            }
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R1_RN5_NN2_DT_AGE_SEX_TNT = M_R1_RN5_NN1_DT_AGE_SEX_TNT._replace(
        description='Reducing the size of the NN to just a Dense-10. Also, '
                    'no dropout for now.',
        model_kwargs={
            'ecg_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.0],
                'batch_norms': [False]
            },
            'ecg_comb_ffnn_kwargs': None,
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': None
        },
        epochs=100,
        batch_size=32,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [305 * 50],
                        'values': [1e-3, 1e-4],
                    }
                },
            }
        },
    )
    M_R1_RN5_NN3_DT_AGE_SEX_TNT = M_R1_RN5_NN1_DT_AGE_SEX_TNT._replace(
        description='Neural network with 100 -> 1 dense layers.',
        model_kwargs={
            'ecg_ffnn_kwargs': {
                'sizes': [100, 1],
                'dropouts': [0.0, 0.0],
                'batch_norms': [False, False]
            },
            'ecg_comb_ffnn_kwargs': None,
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': None
        },
        epochs=100,
        batch_size=32,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [305 * 50],
                        'values': [1e-3, 1e-4],
                    }
                },
            }
        },
    )

    # LOGISTIC REGRESSION USING 1 ECG PROCESSED WITH RESNET + FLAT FEATURES
    M_R1_RN5_LR1_DT_AGE_SEX_TNT = Experiment(
        description='Loads the pre-trained R1_RN5 model and uses it as a '
                    'feature extractor for the input ECG, giving only the '
                    'final scalar as output for each ECG. Add the flat-'
                    'features and plug it all into a logistic regression '
                    'model.',
        model=logistic_regression,
        pre_processor=pre_process_using_xp,
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_RN5',
            'commit': '61fb8038d91ee119a1a889c3c86b27931f1f57b5',
            'which': 'last',
            'final_layer_index': -1
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            }
        },
        epochs=300,
        batch_size=-1,
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1},
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )

    # FFNN USING 2 ECGs PROCESSED WITH CNN4 + FLAT FEATURES
    M_R2_CNN4_NN1 = Experiment(
        description='Loads the pre-trained R1_CNN4 model and uses it as a '
                    'feature extractor for the two input ECGs. The model '
                    'itself is a simple feed-forward neural network with '
                    'a single hidden layer of size 100.',
        model=ffnn,
        model_kwargs={
            'ecg_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.3],
                'batch_norms': [False]
            }
        },
        pre_processor=pre_process_using_xp,
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_CNN4',
            'commit': 'eb783dc3ab36f554b194cffe463919620d123496',
            'final_layer_index': -3
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1']
            },
        },
        class_weight={0: 1, 1: 10},
        epochs=200,
        batch_size=64,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [153 * 100],
                        'values': [1e-4, 1e-5],
                    }
                },
            }
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R2_CNN4_NN1_DT = M_R2_CNN4_NN1._replace(
        description='Use pretrained CNN as feature extractor for ECGs. Feed '
                    'into dense-100 layer, then concatenate time since last '
                    'ecg.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt']
            },
        },
    )
    M_R2_CNN4_NN1_AGE = M_R2_CNN4_NN1._replace(
        description='Use pretrained CNN as feature extractor for ECGs. Feed '
                    'into dense-100 layer, then concatenate age.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['age']
            },
        },
    )
    M_R2_CNN4_NN1_SEX = M_R2_CNN4_NN1._replace(
        description='Use pretrained CNN as feature extractor for ECGs. Feed '
                    'into dense-100 layer, then concatenate sex.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['male']
            },
        },
    )
    M_R2_CNN4_NN1_TNT = M_R2_CNN4_NN1._replace(
        description='Use pretrained CNN as feature extractor for ECGs. Feed '
                    'into dense-100 layer, then concatenate tnt.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['tnt_1']
            },
        },
    )
    M_R2_CNN4_NN1_DT_AGE = M_R2_CNN4_NN1._replace(
        description='Use pretrained CNN as feature extractor for ECGs. Feed '
                    'into dense-100 layer, then concatenate time since last '
                    'ecg and age.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt', 'age']
            },
        },
    )
    M_R2_CNN4_NN1_DT_AGE_SEX = M_R2_CNN4_NN1._replace(
        description='Use pretrained CNN as feature extractor for ECGs. Feed '
                    'into dense-100 layer, then concatenate time since last '
                    'ecg, age and sex',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt', 'age', 'male']
            },
        },
    )
    M_R2_CNN4_NN1_DT_AGE_SEX_TNT = M_R2_CNN4_NN1._replace(
        description='Use pretrained CNN as feature extractor for ECGs. Feed '
                    'into dense-100 layer, then concatenate time since last '
                    'ecg, age, sex and tnt.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
        },
    )

    # RF USING 2 ECGs PROCESSED WITH CNN4
    M_R2_CNN4_RF1 = M_R2_CNN4_NN1._replace(
        description='Try Random Forest instead.',
        model=RandomForestClassifier,
        model_kwargs={
            'n_estimators': 1000,
        },
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_CNN4',
            'commit': 'eb783dc3ab36f554b194cffe463919620d123496',
            'final_layer_index': -3
        },
        building_model_requires_development_data=False,
    )

    # LOGISTIC REGRESSION USING 2 ECGs PROCESSED WITH CNN4 + FLAT FEATURES
    M_R2_CNN4_LR1_DT = Experiment(
        description='Logistic regression, 2 ECGs + dt vs MACE 30',
        model=logistic_regression,
        pre_processor=pre_process_using_xp,
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_CNN4',
            'commit': 'eb783dc3ab36f554b194cffe463919620d123496',
            'final_layer_index': -1
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt']
            },
        },
        epochs=300,
        batch_size=-1,
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1},
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R2_CNN4_LR1_AGE = M_R2_CNN4_LR1_DT._replace(
        description='Logistic regression, 2 ECGs + age vs MACE 30',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['age']
            },
        },
    )
    M_R2_CNN4_LR1_SEX = M_R2_CNN4_LR1_DT._replace(
        description='Logistic regression, 2 ECGs + sex vs MACE 30',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['male']
            },
        },
    )
    M_R2_CNN4_LR1_TNT = M_R2_CNN4_LR1_DT._replace(
        description='Logistic regression, 2 ECGs + tnt vs MACE 30',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['tnt_1']
            },
        },
    )
    M_R2_CNN4_LR1_DT_AGE = M_R2_CNN4_LR1_DT._replace(
        description='Logistic regression, 2 ECGs + dt + age vs MACE 30',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt', 'age']
            },
        },
    )
    M_R2_CNN4_LR1_DT_AGE_SEX = M_R2_CNN4_LR1_DT._replace(
        description='Logistic regression, 2 ECGs + dt + age + sex vs MACE 30',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt', 'age', 'male']
            },
        },
    )
    M_R2_CNN4_LR1_DT_AGE_SEX_TNT = M_R2_CNN4_LR1_DT._replace(
        description='Logistic regression, 2 ECGs + all flat features vs '
                    'MACE 30',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
        },
    )

    # LOGISTIC REGRESSION USING 2 ECGs PROCESSED WITH RESNET + FLAT FEATURES
    M_R2_RN5_LR1_DT_AGE_SEX_TNT = M_R1_RN5_LR1_DT_AGE_SEX_TNT._replace(
        description='Process both input ECGs with the pre-trained ResNet, '
                    'using only the predictions for each as input, together '
                    'with the flat-features.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            }
        },
    )

    # FFNN USING 2 ECGs PROCESSED WITH RESNET + FLAT FEATURES
    M_R2_RN5_NN1 = Experiment(
        description='Loads the pre-trained R1_RN5 model and uses it as a '
                    'feature extractor for the two input ECGs. The model '
                    'itself is a simple feed-forward neural network with '
                    'a single hidden layer of size 100.',
        model=ffnn,
        model_kwargs={
            'final_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.3],
                'batch_norms': [False]
            }
        },
        pre_processor=pre_process_using_xp,
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_RN5',
            'commit': '61fb8038d91ee119a1a889c3c86b27931f1f57b5',
            'which': 'last',
            'final_layer_index': -3
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            }
        },
        epochs=200,
        batch_size=32,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [305 * 100],
                        'values': [1e-3, 1e-4],
                    }
                },
            }
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R2_RN5_NN1_DT_AGE_SEX_TNT = M_R2_RN5_NN1._replace(
        description='Use two ECGs this time.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            }
        },
    )

    # RF USING 2 ECGs PROCESSED WITH RESNET
    M_R2_RN5_RF1 = M_R2_RN5_NN1._replace(
        description='Pre-process the two input ECGs using pretrained ResNet, '
                    'concatenate the result and feed it into a Random Forest '
                    'classifier.',
        model=RandomForestClassifier,
        model_kwargs={
            'n_estimators': 1000,
        },
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_RN5',
            'commit': '61fb8038d91ee119a1a889c3c86b27931f1f57b5',
            'which': 'last',
            'final_layer_index': -3
        },
        building_model_requires_development_data=False,
    )

    # AMI-30
    AMI_LR1_TNT = Experiment(
        description='Logistic regression, ami vs tnt',
        model=logistic_regression,
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'flat_features': ['tnt_1']
            },
            'labels': {
                'target': 'ami30'
            }
        },
        epochs=300,
        batch_size=-1,
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1},
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    AMI_LR1_DT_AGE_SEX_TNT = AMI_LR1_TNT._replace(
        description='Logistic regression, ami vs tnt + dt + age + sex',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
            'labels': {
                'target': 'ami30'
            }
        }
    )

    AMI_R1_CNN2 = Experiment(
        description='Predicting AMI-30 using single raw ECG in a simple '
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
                'down_sample': True
            },
            'final_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.3],
                'batch_norms': [False]
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
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0']
            },
            'labels': {
                'target': 'ami30'
            }
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    AMI_R1_CNN4 = Experiment(
        description='Predicting AMI-30 with CNN4 and only 1 ECG input.',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': True,
                'num_layers': 2,
                'dropout': 0.5,
                'filter_first': 32,
                'filter_last': 32,
                'kernel_first': 16,
                'kernel_last': 16,
                'pool_size': 16,
                'batch_norm': True,
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.5],
                'batch_norms': [False]
            },
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0']
            },
            'labels': {
                'target': 'ami30'
            }
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [153*20, 153*40, 153*150],
                        'values': [1e-3, 1e-4, 1e-5, 1e-6],
                    }
                },
            }
        },
        class_weight={0: 1, 1: 10.7},
        epochs=200,
        batch_size=64,
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )

    AMIr_R1_CNN4 = AMI_R1_CNN4._replace(
        description='Try shuffling the development data instead of splitting '
                    'chronologically.',
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'random_state': 123,
            'test_size': 1 / 3
        },
    )

    AMI_R1_CNN4_TNT = AMI_R1_CNN4._replace(
        description='Predicting AMI-30 with CNN4 and 1 ECG input + TnT',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['tnt_1']
            },
            'labels': {
                'target': 'ami30'
            }
        },
    )
    AMI_R1_CNN4_DT_AGE_SEX_TNT = AMI_R1_CNN4._replace(
        description='Predicting AMI-30 with CNN4 and 1 ECG input + flat-'
                    'features',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
            'labels': {
                'target': 'ami30'
            }
        },
    )

    AMI_R1_AB1 = Experiment(
        description='Predicting AMI-30 using Björkelund et al, except only '
                    '1 ECG input and nothing else.',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': False,
                'num_layers': 3,
                'dropouts': [0.0, 0.3, 0.0],
                'kernels': [64, 16, 16],
                'filters': [64, 16, 8],
                'weight_decays': [1e-4, 1e-3, 1e-4],
                'pool_sizes': [32, 4, 8],
                'batch_norm': False,
            },
            'ecg_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.0],
                'batch_norms': [False]
            },
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.5],
                'batch_norms': [False]
            }
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0']
            },
            'labels': {
                'target': 'ami30'
            }
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 3e-3}
        },
        epochs=200,
        batch_size=64,
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    AMI_R1_AB1_DT_AGE_SEX_TNT = AMI_R1_AB1._replace(
        description='Predicting AMI with AB1, using 1 ECG plus the flat-'
                    'features',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
            'labels': {
                'target': 'ami30'
            }
        },
    )
    AMI_R2_AB1_DT_AGE_SEX_TNT = AMI_R1_AB1._replace(
        description='Predicting AMI with AB1, using 2 ECGs plus the flat-'
                    'features',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
            'labels': {
                'target': 'ami30'
            }
        },
    )

    AMI_R1_RN5 = Experiment(
        description="Pretrained ResNet architecture from Ribeiro et al, "
                    "predicting AMI-30.",
        model=load_ribeiro_model,
        model_kwargs={},
        epochs=200,
        batch_size=32,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [305 * 20, 305 * 100],
                        'values': [1e-3, 1e-4, 1e-5],
                    }
                },
            }
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            },
            'labels': {
                'target': 'ami30'
            }
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
