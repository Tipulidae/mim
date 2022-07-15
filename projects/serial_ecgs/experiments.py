from enum import Enum

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit

from mim.experiments import hyper_parameter as hp
from mim.experiments.experiments import Experiment
from mim.experiments.hyper_experiments import HyperExperiment
from mim.experiments.search_strategies import RandomSearch, Hyperband
from mim.experiments.extractor import sklearn_process
from mim.cross_validation import ChronologicalSplit
from projects.serial_ecgs.extractor import EscTrop
from projects.serial_ecgs.models import (
    ecg_cnn,
    ffnn,
    logistic_regression,
    logistic_regression_ab,
    pretrained_resnet
)


class ESCT(Experiment, Enum):
    # LOGISTIC REGRESSION USING FLAT FEATURES:
    M_FF_LR1 = Experiment(
        description='Predicting MACE using flat-features. Tensorflow logistic '
                    'regression model. ',
        model=logistic_regression,
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'flat_features': ['log_tnt_1', 'male', 'age', 'log_dt']
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
    M_FF_LR2 = Experiment(
        description='Scikit-learns logistic regression model, mace vs flat '
                    'features.',
        model=LogisticRegression,
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'flat_features': ['log_tnt_1', 'male', 'age', 'log_dt']
            },
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'flat_features': {
                'processor': StandardScaler,
            }
        },
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_FF_LR3 = Experiment(
        description="Anders Björkelund's logistic regression model.",
        model=logistic_regression_ab,
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'flat_features': ['log_tnt_1', 'male', 'age', 'log_dt']
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

    # LOGISTIC REGRESSION USING "FORBERG"-FEATURES
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
                'forberg': ['ecg_0', 'combine']
            },
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'forberg_features': {
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
                'forberg': ['ecg_0', 'diff', 'combine']
            },
        },
        pre_processor_kwargs={
            'forberg_features': {
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
    )
    M_F1_FF_LR2 = M_F1_LR2._replace(
        description='Scikit-learn logistic regression model, mace vs '
                    'features from Forberg et al. Features are normalized '
                    'and then reduced in dimension by PCA.',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male', 'log_tnt_1'],
                'forberg': ['ecg_0', 'combine']
            },
        },
        pre_processor_kwargs={
            'forberg_features': {
                'processor': 'Pipeline',
                'steps': [
                    ('scaler', StandardScaler, {}),
                    ('pca', PCA, {
                        'n_components': 2,
                        'whiten': False,
                        'random_state': 42
                    })
                ]
            },
            'flat_features': {
                'processor': StandardScaler,
            }
        },
    )
    M_F2_FF_LR2 = M_F1_LR2._replace(
        description='Scikit-learn logistic regression model, mace vs '
                    'features from Forberg et al. Features are normalized '
                    'and then reduced in dimension by PCA. I tried a bunch '
                    'of settings for dimension, ~150 gave the best AUC.',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male', 'log_tnt_1'],
                'forberg': ['ecg_0', 'diff', 'combine']
            },
        },
        pre_processor_kwargs={
            'forberg_features': {
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
            'flat_features': {
                'processor': StandardScaler,
            }
        },
    )

    # FFNN USING FORBERG-FEATURES
    M_F1_NN1 = Experiment(
        description='Best iteration (xp_184) from M_F1_NN_RS',
        model=ffnn,
        model_kwargs={
            'ecg_ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.3],
                'batch_norms': [False],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.001],
                'bias_regularizers': [0.1]
            },
            'ecg_combiner': None,
            'ecg_comb_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.0],
                'batch_norms': [False],
                'activity_regularizers': [0.0001],
                'kernel_regularizers': [0.01],
                'bias_regularizers': [0.001]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': None,
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'forberg': ['ecg_0']
            },
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'forberg_ecg_0': {'processor': StandardScaler},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.0001,
            }
        },
        epochs=100,
        batch_size=64,
        cv=ChronologicalSplit,
        cv_kwargs={
            'test_size': 1 / 3
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc'],
    )
    M_F2_NN2 = Experiment(
        description='Best iteration (xp_324) from M_F2_NN_RS',
        model=ffnn,
        model_kwargs={
            'ecg_ffnn_kwargs': {
                'sizes': [200],
                'dropouts': [0.3],
                'batch_norms': [False],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.0001],
                'bias_regularizers': [0.01]
            },
            'ecg_combiner': 'concatenate',
            'ecg_comb_ffnn_kwargs': {
                'sizes': [6],
                'dropouts': [0.0],
                'batch_norms': [False],
                'activity_regularizers': [0.001],
                'kernel_regularizers': [0.1],
                'bias_regularizers': [0.0]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': None,
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'forberg': ['ecg_0', 'ecg_1']
            },
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'forberg_ecg_0': {'processor': StandardScaler},
            'forberg_ecg_1': {'processor': StandardScaler},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.0001,
            }
        },
        epochs=100,
        batch_size=64,
        cv=ChronologicalSplit,
        cv_kwargs={
            'test_size': 1 / 3
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc'],
    )
    M_F1_FF_NN3 = Experiment(
        description='Best iteration (xp_353) from M_F1_FF_NN_RS',
        model=ffnn,
        model_kwargs={
            'ecg_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.2],
                'batch_norms': [True],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.1],
                'bias_regularizers': [0.001]
            },
            'ecg_combiner': None,
            'ecg_comb_ffnn_kwargs': {
                'sizes': [20],
                'dropouts': [0.3],
                'batch_norms': [False],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.001],
                'bias_regularizers': [0.0]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.0],
                'batch_norms': [True],
                'activity_regularizers': [0.0001],
                'kernel_regularizers': [0.1],
                'bias_regularizers': [0.01]
            },
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt'],
                'forberg': ['ecg_0']
            },
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'flat_features': {'processor': StandardScaler},
            'forberg_ecg_0': {'processor': StandardScaler},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.0003,
            }
        },
        epochs=100,
        batch_size=64,
        cv=ChronologicalSplit,
        cv_kwargs={
            'test_size': 1 / 3
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc'],
    )
    M_F2_FF_NN4 = Experiment(
        description='Best iteration (xp_224) from M_F2_FF_NN_RS',
        model=ffnn,
        model_kwargs={
            'ecg_ffnn_kwargs': {
                'sizes': [25],
                'dropouts': [0.3],
                'batch_norms': [True],
                'activity_regularizers': [0.001],
                'kernel_regularizers': [0.1],
                'bias_regularizers': [0.1]
            },
            'ecg_combiner': 'concatenate',
            'ecg_comb_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.0],
                'batch_norms': [True],
                'activity_regularizers': [0.0001],
                'kernel_regularizers': [0.1],
                'bias_regularizers': [0.1]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.3],
                'batch_norms': [False],
                'activity_regularizers': [0.0],
                'kernel_regularizers': [0.0],
                'bias_regularizers': [0.001]
            },
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt'],
                'forberg': ['ecg_0', 'ecg_1']
            },
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'flat_features': {'processor': StandardScaler},
            'forberg_ecg_0': {'processor': StandardScaler},
            'forberg_ecg_1': {'processor': StandardScaler},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.0003,
            }
        },
        epochs=100,
        batch_size=64,
        cv=ChronologicalSplit,
        cv_kwargs={
            'test_size': 1 / 3
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc'],
    )
    M_F2_FF_NN5 = Experiment(
        description='Second best iteration (xp_63) from M_F2_FF_NN_RS',
        model=ffnn,
        model_kwargs={
            'ecg_ffnn_kwargs': {
                'sizes': [200],
                'dropouts': [0.4],
                'batch_norms': [False],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.01],
                'bias_regularizers': [0.0]
            },
            'ecg_combiner': 'difference',
            'ecg_comb_ffnn_kwargs': {
                'sizes': [6],
                'dropouts': [0.1],
                'batch_norms': [True],
                'activity_regularizers': [0.0],
                'kernel_regularizers': [0.001],
                'bias_regularizers': [0.1]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.4],
                'batch_norms': [True],
                'activity_regularizers': [0.001],
                'kernel_regularizers': [0.0001],
                'bias_regularizers': [0.001]
            },
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt'],
                'forberg': ['ecg_0', 'ecg_1']
            },
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'flat_features': {'processor': StandardScaler},
            'forberg_ecg_0': {'processor': StandardScaler},
            'forberg_ecg_1': {'processor': StandardScaler},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.0003,
            }
        },
        epochs=100,
        batch_size=64,
        cv=ChronologicalSplit,
        cv_kwargs={
            'test_size': 1 / 3
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc'],
    )
    M_F2_FF_NN6 = Experiment(
        description='Third best iteration (xp_208) from M_F2_FF_NN_RS',
        model=ffnn,
        model_kwargs={
            'ecg_ffnn_kwargs': {
                'sizes': [200],
                'dropouts': [0.3],
                'batch_norms': [False],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.001],
                'bias_regularizers': [0.1]
            },
            'ecg_combiner': 'difference',
            'ecg_comb_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.4],
                'batch_norms': [True],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.1],
                'bias_regularizers': [0.1]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.1],
                'batch_norms': [True],
                'activity_regularizers': [0.0],
                'kernel_regularizers': [0.0],
                'bias_regularizers': [0.0001]
            },
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt'],
                'forberg': ['ecg_0', 'ecg_1']
            },
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'flat_features': {'processor': StandardScaler},
            'forberg_ecg_0': {'processor': StandardScaler},
            'forberg_ecg_1': {'processor': StandardScaler},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.001,
            }
        },
        epochs=100,
        batch_size=64,
        cv=ChronologicalSplit,
        cv_kwargs={
            'test_size': 1 / 3
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc'],
    )

    # CNN1, (2nd) best random-search model for 1 ECG
    M_R1_CNN1 = Experiment(
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
        epochs=100,
        batch_size=64,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    M_R1_CNN1b = M_R1_CNN1._replace(
        description='Replaces the final layer with something smaller.',
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
                'sizes': [10],
                'dropouts': [0.3],
                'batch_norms': [False]
            }
        },
    )
    # CNN2, best random-search model for 2 ECGs
    M_R2_CNN2 = Experiment(
        description='Uses xp_26 from M_R2_CNN_RS, which was best in terms '
                    'of AUC.',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'down_sample': True,
                'dropouts': [0.0, 0.2],
                'pool_size': 19,
                'filter_first': 32,
                'filter_last': 64,
                'kernel_first': 49,
                'kernel_last': 17,
                'batch_norms': [False, True],
                'weight_decays': [0.001, 0.1],
            },
            'ecg_ffnn_kwargs': None,
            'ecg_comb_ffnn_kwargs': {
                'sizes': [20],
                'dropouts': [0.2],
                'batch_norms': [False]
            },
            'ecg_combiner': 'concatenate',
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [20],
                'dropouts': [0.0],
                'batch_norms': [False]
            }
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.00001}
        },
        epochs=100,
        batch_size=64,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    M_R2_CNN2b = M_R2_CNN2._replace(
        description='Changes the random state to the exact same as in the '
                    'RS experiment...',
        random_state=634181651
    )

    # CNN3, best random-search model for 1 ECG + flat-features
    M_R1_CNN3 = Experiment(
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
        epochs=100,
        batch_size=64,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1/3},
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    M_R1_FF_CNN3 = M_R1_CNN3._replace(
        description='Uses the CNN3 (or xp_379) model with flat-features, as '
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
    M_R2_FF_CNN3 = M_R1_CNN3._replace(
        description='Uses the CNN3 (or xp_379) model, except with 2 ECGs '
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

    # CNN4, best random-search model for 2 ECGs + flat-features
    M_R1_CNN4 = Experiment(
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
        epochs=100,
        batch_size=64,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1/3},
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    M_R1_FF_CNN4 = M_R1_CNN4._replace(
        description='Uses the CNN4 model with 1 ECG and flat-features. ',
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
        }
    )
    M_R2_FF_CNN4 = M_R1_CNN4._replace(
        description='Uses the CNN4 model with 2 ECGs and flat-features, '
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
    M_R2_FF_CNN4b = Experiment(
        description='Adjusts the network slightly to not use such a large '
                    'final dense layer.',
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
                'sizes': [10],
                'dropouts': [0.5],
                'batch_norms': [False]
            }
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
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

    # CNN5, second best random-search model for 1 ECG + ff
    M_R1_FF_CNN5 = Experiment(
        description='Uses xp_382 from M_R1_FF_CNN_RS, which was the second '
                    'best in terms of both AUC and rule-out. ',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'down_sample': True,
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
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
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

    # CNN6, second best random-search model for 2 ECGs + ff
    M_R2_FF_CNN6 = Experiment(
        description='Second best model using 2ECGs + ff (xp_245).',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': True,
                'num_layers': 4,
                'dropouts': [0.3, 0.0, 0.5, 0.3],
                'pool_size': 6,
                'filter_first': 20,
                'filter_last': 20,
                'kernel_first': 21,
                'kernel_last': 17,
                'batch_norms': [True, False, False, False],
                'weight_decays': [0.01, 0.001, 0.01, 0.01],
            },
            'ecg_combiner': 'difference',
            'ecg_ffnn_kwargs': None,
            'ecg_comb_ffnn_kwargs': {
                'sizes': [20],
                'dropouts': [0.5],
                'batch_norms': [False]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.5],
                'batch_norms': [False]
            }
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.0003}
        },
        epochs=100,
        batch_size=64,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1/3},
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    M_R2_FF_CNN6b = Experiment(
        description='Model adjusted slightly to reduce the final dense '
                    'size.',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': True,
                'num_layers': 4,
                'dropouts': [0.3, 0.0, 0.5, 0.3],
                'pool_size': 6,
                'filter_first': 20,
                'filter_last': 20,
                'kernel_first': 21,
                'kernel_last': 17,
                'batch_norms': [True, False, False, False],
                'weight_decays': [0.01, 0.001, 0.01, 0.01],
            },
            'ecg_combiner': 'concatenate',
            'ecg_ffnn_kwargs': None,
            'ecg_comb_ffnn_kwargs': {
                'sizes': [20],
                'dropouts': [0.5],
                'batch_norms': [False]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [20],
                'dropouts': [0.5],
                'batch_norms': [False]
            }
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.0003}
        },
        epochs=100,
        batch_size=64,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1/3},
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )

    # CNN7, best random-search model for 1 ECG
    M_R1_CNN7 = Experiment(
        description='Uses xp_261 from M_R1_CNN_RS, which was the '
                    'best in terms of AUC, but learning curve might have '
                    'been a fluke. Testing it here anyway to be sure.',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 3,
                'down_sample': True,
                'dropouts': [0.3, 0.4, 0.3],
                'pool_size': 10,
                'filter_first': 36,
                'filter_last': 32,
                'kernel_first': 49,
                'kernel_last': 37,
                'batch_norms': [True, False, True],
                'weight_decays': [0.001, 0.001, 0.01],
            },
            'ecg_ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.0],
                'batch_norms': [False]
            },
            'ecg_comb_ffnn_kwargs': None,
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [10],
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
        cv_kwargs={'test_size': 1 / 3},
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )

    # RN1, 1 ECG
    M_R1_RN1 = Experiment(
        description='xp_141 from R1_RN_RS',
        model=pretrained_resnet,
        model_kwargs={
            'freeze_resnet': False,
            'ecg_ffnn_kwargs': {
                'sizes': [200, 100],
                'dropouts': [0.0, 0.1],
                'batch_norms': [False, True],
                'activity_regularizers': [0.0, 0.0001],
                'kernel_regularizers': [0.01, 0.0],
                'bias_regularizers': [0.0001, 0.1]
            },
            'ecg_combiner': 'difference',
            'ecg_comb_ffnn_kwargs': {
                'sizes': [6],
                'dropouts': [0.0],
                'batch_norms': [False],
                'activity_regularizers': [0.0001],
                'kernel_regularizers': [0.1],
                'bias_regularizers': [0.001]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': None,
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            }
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.0003}
        },
        epochs=50,
        batch_size=32,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )

    # RN2, 2 ECGs
    M_R2_RN2 = Experiment(
        description='xp_133 from R1_RN_RS, 2nd best model.',
        model=pretrained_resnet,
        model_kwargs={
            'freeze_resnet': False,
            'ecg_ffnn_kwargs': {
                'sizes': [200, 50],
                'dropouts': [0.2, 0.0],
                'batch_norms': [False, False],
                'activity_regularizers': [0.001, 0.00001],
                'kernel_regularizers': [0.01, 0.01],
                'bias_regularizers': [0.01, 0.001]
            },
            'ecg_combiner': 'difference',
            'ecg_comb_ffnn_kwargs': {
                'sizes': [20],
                'dropouts': [0.3],
                'batch_norms': [False],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.01],
                'bias_regularizers': [0.01]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': None,
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            }
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [5490],
                        'values': [0.001, 0.00001]
                    }
                }
            }
        },
        epochs=50,
        batch_size=32,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )

    # RN3, 1 ECG+ff
    M_R1_FF_RN3 = Experiment(
        description='xp_166 from R1_FF_RN_RS, 2nd best model',
        model=pretrained_resnet,
        model_kwargs={
            'freeze_resnet': False,
            'ecg_ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.5],
                'batch_norms': [False],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.01],
                'bias_regularizers': [0.0],
            },
            'ecg_combiner': 'concatenate',
            'ecg_comb_ffnn_kwargs': {
                'sizes': [6],
                'dropouts': [0.0],
                'batch_norms': [True],
                'activity_regularizers': [0.0001],
                'kernel_regularizers': [0.01],
                'bias_regularizers': [0.1]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.2],
                'batch_norms': [True],
                'activity_regularizers': [0.0001],
                'kernel_regularizers': [0.0],
                'bias_regularizers': [0.0001],
            },
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            }
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [7320],
                        'values': [0.0003, 0.00003]
                    }
                },
            }
        },
        epochs=50,
        batch_size=32,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )

    # RN4, 2 ECGs+ff
    M_R2_FF_RN4 = Experiment(
        description='xp_24 from R2_FF_RN_RS',
        model=pretrained_resnet,
        model_kwargs={
            'freeze_resnet': False,
            'ecg_ffnn_kwargs': {
                'sizes': [25],
                'dropouts': [0.2],
                'batch_norms': [False],
                'activity_regularizers': [0.01],
                'kernel_regularizers': [0.001],
                'bias_regularizers': [0.001]
            },
            'ecg_combiner': 'difference',
            'ecg_comb_ffnn_kwargs': {
                'sizes': [6],
                'dropouts': [0.0],
                'batch_norms': [False],
                'activity_regularizers': [0.0001],
                'kernel_regularizers': [0.0001],
                'bias_regularizers': [0.0001]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.2],
                'batch_norms': [True],
                'activity_regularizers': [0.0],
                'kernel_regularizers': [0.0001],
                'bias_regularizers': [0.01],
            },
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            }
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [10065],
                        'values': [0.003, 0.00001]
                    }
                },
            }
        },
        epochs=50,
        batch_size=32,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )

    # ENSEMBLES
    # NN1 ensemble
    M_F1_NN1_r1 = M_F1_NN1._replace(
        description='Same as M_F1_NN1, but with a different random seed.',
        random_state=1001,
    )
    M_F1_NN1_r2 = M_F1_NN1._replace(
        description='Same as M_F1_NN1, but with a different random seed.',
        random_state=1002,
    )
    M_F1_NN1_r3 = M_F1_NN1._replace(
        description='Same as M_F1_NN1, but with a different random seed.',
        random_state=1003,
    )
    M_F1_NN1_r4 = M_F1_NN1._replace(
        description='Same as M_F1_NN1, but with a different random seed.',
        random_state=1004,
    )
    M_F1_NN1_r5 = M_F1_NN1._replace(
        description='Same as M_F1_NN1, but with a different random seed.',
        random_state=1005,
    )
    M_F1_NN1_r6 = M_F1_NN1._replace(
        description='Same as M_F1_NN1, but with a different random seed.',
        random_state=1006,
    )
    M_F1_NN1_r7 = M_F1_NN1._replace(
        description='Same as M_F1_NN1, but with a different random seed.',
        random_state=1007,
    )
    M_F1_NN1_r8 = M_F1_NN1._replace(
        description='Same as M_F1_NN1, but with a different random seed.',
        random_state=1008,
    )
    M_F1_NN1_r9 = M_F1_NN1._replace(
        description='Same as M_F1_NN1, but with a different random seed.',
        random_state=1009,
    )

    # NN2 ensemble
    M_F2_NN2_r1 = M_F2_NN2._replace(
        description='Same as M_F2_NN2, but with a different random seed.',
        random_state=1011,
    )
    M_F2_NN2_r2 = M_F2_NN2._replace(
        description='Same as M_F2_NN2, but with a different random seed.',
        random_state=1012,
    )
    M_F2_NN2_r3 = M_F2_NN2._replace(
        description='Same as M_F2_NN2, but with a different random seed.',
        random_state=1013,
    )
    M_F2_NN2_r4 = M_F2_NN2._replace(
        description='Same as M_F2_NN2, but with a different random seed.',
        random_state=1014,
    )
    M_F2_NN2_r5 = M_F2_NN2._replace(
        description='Same as M_F2_NN2, but with a different random seed.',
        random_state=1015,
    )
    M_F2_NN2_r6 = M_F2_NN2._replace(
        description='Same as M_F2_NN2, but with a different random seed.',
        random_state=1016,
    )
    M_F2_NN2_r7 = M_F2_NN2._replace(
        description='Same as M_F2_NN2, but with a different random seed.',
        random_state=1017,
    )
    M_F2_NN2_r8 = M_F2_NN2._replace(
        description='Same as M_F2_NN2, but with a different random seed.',
        random_state=1018,
    )
    M_F2_NN2_r9 = M_F2_NN2._replace(
        description='Same as M_F2_NN2, but with a different random seed.',
        random_state=1019,
    )

    # NN3 ensemble
    M_F1_FF_NN3_r1 = M_F1_FF_NN3._replace(
        description='Same as M_F1_FF_NN3, but with a different random seed.',
        random_state=1021,
    )
    M_F1_FF_NN3_r2 = M_F1_FF_NN3._replace(
        description='Same as M_F1_FF_NN3, but with a different random seed.',
        random_state=1022,
    )
    M_F1_FF_NN3_r3 = M_F1_FF_NN3._replace(
        description='Same as M_F1_FF_NN3, but with a different random seed.',
        random_state=1023,
    )
    M_F1_FF_NN3_r4 = M_F1_FF_NN3._replace(
        description='Same as M_F1_FF_NN3, but with a different random seed.',
        random_state=1024,
    )
    M_F1_FF_NN3_r5 = M_F1_FF_NN3._replace(
        description='Same as M_F1_FF_NN3, but with a different random seed.',
        random_state=1025,
    )
    M_F1_FF_NN3_r6 = M_F1_FF_NN3._replace(
        description='Same as M_F1_FF_NN3, but with a different random seed.',
        random_state=1026,
    )
    M_F1_FF_NN3_r7 = M_F1_FF_NN3._replace(
        description='Same as M_F1_FF_NN3, but with a different random seed.',
        random_state=1027,
    )
    M_F1_FF_NN3_r8 = M_F1_FF_NN3._replace(
        description='Same as M_F1_FF_NN3, but with a different random seed.',
        random_state=1028,
    )
    M_F1_FF_NN3_r9 = M_F1_FF_NN3._replace(
        description='Same as M_F1_FF_NN3, but with a different random seed.',
        random_state=1029,
    )

    # NN4 ensemble
    M_F2_FF_NN4_r1 = M_F2_FF_NN4._replace(
        description='Same as M_F2_FF_NN4, but with a different random seed.',
        random_state=1031,
    )
    M_F2_FF_NN4_r2 = M_F2_FF_NN4._replace(
        description='Same as M_F2_FF_NN4, but with a different random seed.',
        random_state=1032,
    )
    M_F2_FF_NN4_r3 = M_F2_FF_NN4._replace(
        description='Same as M_F2_FF_NN4, but with a different random seed.',
        random_state=1033,
    )
    M_F2_FF_NN4_r4 = M_F2_FF_NN4._replace(
        description='Same as M_F2_FF_NN4, but with a different random seed.',
        random_state=1034,
    )
    M_F2_FF_NN4_r5 = M_F2_FF_NN4._replace(
        description='Same as M_F2_FF_NN4, but with a different random seed.',
        random_state=1035,
    )
    M_F2_FF_NN4_r6 = M_F2_FF_NN4._replace(
        description='Same as M_F2_FF_NN4, but with a different random seed.',
        random_state=1036,
    )
    M_F2_FF_NN4_r7 = M_F2_FF_NN4._replace(
        description='Same as M_F2_FF_NN4, but with a different random seed.',
        random_state=1037,
    )
    M_F2_FF_NN4_r8 = M_F2_FF_NN4._replace(
        description='Same as M_F2_FF_NN4, but with a different random seed.',
        random_state=1038,
    )
    M_F2_FF_NN4_r9 = M_F2_FF_NN4._replace(
        description='Same as M_F2_FF_NN4, but with a different random seed.',
        random_state=1039,
    )

    # NN5 ensemble (2nd best on 2ECGs + ff)
    M_F2_FF_NN5_r1 = M_F2_FF_NN5._replace(
        description='Same as M_F2_FF_NN5, but with a different random seed.',
        random_state=1031,
    )
    M_F2_FF_NN5_r2 = M_F2_FF_NN5._replace(
        description='Same as M_F2_FF_NN5, but with a different random seed.',
        random_state=1032,
    )
    M_F2_FF_NN5_r3 = M_F2_FF_NN5._replace(
        description='Same as M_F2_FF_NN5, but with a different random seed.',
        random_state=1033,
    )
    M_F2_FF_NN5_r4 = M_F2_FF_NN5._replace(
        description='Same as M_F2_FF_NN5, but with a different random seed.',
        random_state=1034,
    )
    M_F2_FF_NN5_r5 = M_F2_FF_NN5._replace(
        description='Same as M_F2_FF_NN5, but with a different random seed.',
        random_state=1035,
    )
    M_F2_FF_NN5_r6 = M_F2_FF_NN5._replace(
        description='Same as M_F2_FF_NN5, but with a different random seed.',
        random_state=1036,
    )
    M_F2_FF_NN5_r7 = M_F2_FF_NN5._replace(
        description='Same as M_F2_FF_NN5, but with a different random seed.',
        random_state=1037,
    )
    M_F2_FF_NN5_r8 = M_F2_FF_NN5._replace(
        description='Same as M_F2_FF_NN5, but with a different random seed.',
        random_state=1038,
    )
    M_F2_FF_NN5_r9 = M_F2_FF_NN5._replace(
        description='Same as M_F2_FF_NN5, but with a different random seed.',
        random_state=1039,
    )

    # NN6 ensemble (3rd best on 2ECGs + ff)
    M_F2_FF_NN6_r1 = M_F2_FF_NN6._replace(
        description='Same as M_F2_FF_NN6, but with a different random seed.',
        random_state=1031,
    )
    M_F2_FF_NN6_r2 = M_F2_FF_NN6._replace(
        description='Same as M_F2_FF_NN6, but with a different random seed.',
        random_state=1032,
    )
    M_F2_FF_NN6_r3 = M_F2_FF_NN6._replace(
        description='Same as M_F2_FF_NN6, but with a different random seed.',
        random_state=1033,
    )
    M_F2_FF_NN6_r4 = M_F2_FF_NN6._replace(
        description='Same as M_F2_FF_NN6, but with a different random seed.',
        random_state=1034,
    )
    M_F2_FF_NN6_r5 = M_F2_FF_NN6._replace(
        description='Same as M_F2_FF_NN6, but with a different random seed.',
        random_state=1035,
    )
    M_F2_FF_NN6_r6 = M_F2_FF_NN6._replace(
        description='Same as M_F2_FF_NN6, but with a different random seed.',
        random_state=1036,
    )
    M_F2_FF_NN6_r7 = M_F2_FF_NN6._replace(
        description='Same as M_F2_FF_NN6, but with a different random seed.',
        random_state=1037,
    )
    M_F2_FF_NN6_r8 = M_F2_FF_NN6._replace(
        description='Same as M_F2_FF_NN6, but with a different random seed.',
        random_state=1038,
    )
    M_F2_FF_NN6_r9 = M_F2_FF_NN6._replace(
        description='Same as M_F2_FF_NN6, but with a different random seed.',
        random_state=1039,
    )

    # CNN1 ensemble
    M_R1_CNN1_r1 = M_R1_CNN1._replace(
        description='Same as M_R1_CNN1, but with a different random seed.',
        random_state=1041,
    )
    M_R1_CNN1_r2 = M_R1_CNN1._replace(
        description='Same as M_R1_CNN1, but with a different random seed.',
        random_state=1042,
    )
    M_R1_CNN1_r3 = M_R1_CNN1._replace(
        description='Same as M_R1_CNN1, but with a different random seed.',
        random_state=1043,
    )
    M_R1_CNN1_r4 = M_R1_CNN1._replace(
        description='Same as M_R1_CNN1, but with a different random seed.',
        random_state=1044,
    )
    M_R1_CNN1_r5 = M_R1_CNN1._replace(
        description='Same as M_R1_CNN1, but with a different random seed.',
        random_state=1045,
    )
    M_R1_CNN1_r6 = M_R1_CNN1._replace(
        description='Same as M_R1_CNN1, but with a different random seed.',
        random_state=1046,
    )
    M_R1_CNN1_r7 = M_R1_CNN1._replace(
        description='Same as M_R1_CNN1, but with a different random seed.',
        random_state=1047,
    )
    M_R1_CNN1_r8 = M_R1_CNN1._replace(
        description='Same as M_R1_CNN1, but with a different random seed.',
        random_state=1048,
    )
    M_R1_CNN1_r9 = M_R1_CNN1._replace(
        description='Same as M_R1_CNN1, but with a different random seed.',
        random_state=1049,
    )

    # CNN2 ensemble
    M_R2_CNN2_r1 = M_R2_CNN2._replace(
        description='Same as M_R2_CNN2, but with a different random seed.',
        random_state=1051,
    )
    M_R2_CNN2_r2 = M_R2_CNN2._replace(
        description='Same as M_R2_CNN2, but with a different random seed.',
        random_state=1052,
    )
    M_R2_CNN2_r3 = M_R2_CNN2._replace(
        description='Same as M_R2_CNN2, but with a different random seed.',
        random_state=1053,
    )
    M_R2_CNN2_r4 = M_R2_CNN2._replace(
        description='Same as M_R2_CNN2, but with a different random seed.',
        random_state=1054,
    )
    M_R2_CNN2_r5 = M_R2_CNN2._replace(
        description='Same as M_R2_CNN2, but with a different random seed.',
        random_state=1055,
    )
    M_R2_CNN2_r6 = M_R2_CNN2._replace(
        description='Same as M_R2_CNN2, but with a different random seed.',
        random_state=1056,
    )
    M_R2_CNN2_r7 = M_R2_CNN2._replace(
        description='Same as M_R2_CNN2, but with a different random seed.',
        random_state=1057,
    )
    M_R2_CNN2_r8 = M_R2_CNN2._replace(
        description='Same as M_R2_CNN2, but with a different random seed.',
        random_state=1058,
    )
    M_R2_CNN2_r9 = M_R2_CNN2._replace(
        description='Same as M_R2_CNN2, but with a different random seed.',
        random_state=1059,
    )

    # CNN3 ensemble
    M_R1_FF_CNN3_r1 = M_R1_FF_CNN3._replace(
        description='Same as M_R1_FF_CNN3, but with a different random seed.',
        random_state=1061,
    )
    M_R1_FF_CNN3_r2 = M_R1_FF_CNN3._replace(
        description='Same as M_R1_FF_CNN3, but with a different random seed.',
        random_state=1062,
    )
    M_R1_FF_CNN3_r3 = M_R1_FF_CNN3._replace(
        description='Same as M_R1_FF_CNN3, but with a different random seed.',
        random_state=1063,
    )
    M_R1_FF_CNN3_r4 = M_R1_FF_CNN3._replace(
        description='Same as M_R1_FF_CNN3, but with a different random seed.',
        random_state=1064,
    )
    M_R1_FF_CNN3_r5 = M_R1_FF_CNN3._replace(
        description='Same as M_R1_FF_CNN3, but with a different random seed.',
        random_state=1065,
    )
    M_R1_FF_CNN3_r6 = M_R1_FF_CNN3._replace(
        description='Same as M_R1_FF_CNN3, but with a different random seed.',
        random_state=1066,
    )
    M_R1_FF_CNN3_r7 = M_R1_FF_CNN3._replace(
        description='Same as M_R1_FF_CNN3, but with a different random seed.',
        random_state=1067,
    )
    M_R1_FF_CNN3_r8 = M_R1_FF_CNN3._replace(
        description='Same as M_R1_FF_CNN3, but with a different random seed.',
        random_state=1068,
    )
    M_R1_FF_CNN3_r9 = M_R1_FF_CNN3._replace(
        description='Same as M_R1_FF_CNN3, but with a different random seed.',
        random_state=1069,
    )

    # CNN4 ENSEMBLE
    M_R2_FF_CNN4_r1 = M_R2_FF_CNN4._replace(
        description='Same as M_R2_FF_CNN4, but with a different random seed.',
        random_state=1071,
    )
    M_R2_FF_CNN4_r2 = M_R2_FF_CNN4._replace(
        description='Same as M_R2_FF_CNN4, but with a different random seed.',
        random_state=1072,
    )
    M_R2_FF_CNN4_r3 = M_R2_FF_CNN4._replace(
        description='Same as M_R2_FF_CNN4, but with a different random seed.',
        random_state=1073,
    )
    M_R2_FF_CNN4_r4 = M_R2_FF_CNN4._replace(
        description='Same as M_R2_FF_CNN4, but with a different random seed.',
        random_state=1074,
    )
    M_R2_FF_CNN4_r5 = M_R2_FF_CNN4._replace(
        description='Same as M_R2_FF_CNN4, but with a different random seed.',
        random_state=1075,
    )
    M_R2_FF_CNN4_r6 = M_R2_FF_CNN4._replace(
        description='Same as M_R2_FF_CNN4, but with a different random seed.',
        random_state=1076,
    )
    M_R2_FF_CNN4_r7 = M_R2_FF_CNN4._replace(
        description='Same as M_R2_FF_CNN4, but with a different random seed.',
        random_state=1077,
    )
    M_R2_FF_CNN4_r8 = M_R2_FF_CNN4._replace(
        description='Same as M_R2_FF_CNN4, but with a different random seed.',
        random_state=1078,
    )
    M_R2_FF_CNN4_r9 = M_R2_FF_CNN4._replace(
        description='Same as M_R2_FF_CNN4, but with a different random seed.',
        random_state=1079,
    )

    # RN1 Ensemble
    M_R1_RN1_r1 = M_R1_RN1._replace(
        description='Same as M_R1_RN1, but with different random seed',
        random_state=1081,
    )
    M_R1_RN1_r2 = M_R1_RN1._replace(
        description='Same as M_R1_RN1, but with different random seed',
        random_state=1082,
    )
    M_R1_RN1_r3 = M_R1_RN1._replace(
        description='Same as M_R1_RN1, but with different random seed',
        random_state=1083,
    )
    M_R1_RN1_r4 = M_R1_RN1._replace(
        description='Same as M_R1_RN1, but with different random seed',
        random_state=1084,
    )
    M_R1_RN1_r5 = M_R1_RN1._replace(
        description='Same as M_R1_RN1, but with different random seed',
        random_state=1085,
    )
    M_R1_RN1_r6 = M_R1_RN1._replace(
        description='Same as M_R1_RN1, but with different random seed',
        random_state=1086,
    )
    M_R1_RN1_r7 = M_R1_RN1._replace(
        description='Same as M_R1_RN1, but with different random seed',
        random_state=1087,
    )
    M_R1_RN1_r8 = M_R1_RN1._replace(
        description='Same as M_R1_RN1, but with different random seed',
        random_state=1088,
    )
    M_R1_RN1_r9 = M_R1_RN1._replace(
        description='Same as M_R1_RN1, but with different random seed',
        random_state=1089,
    )

    # RN2 Ensemble
    M_R2_RN2_r1 = M_R2_RN2._replace(
        description='Same as M_R2_RN2, but with different random seed',
        random_state=1091,
    )
    M_R2_RN2_r2 = M_R2_RN2._replace(
        description='Same as M_R2_RN2, but with different random seed',
        random_state=1092,
    )
    M_R2_RN2_r3 = M_R2_RN2._replace(
        description='Same as M_R2_RN2, but with different random seed',
        random_state=1093,
    )
    M_R2_RN2_r4 = M_R2_RN2._replace(
        description='Same as M_R2_RN2, but with different random seed',
        random_state=1094,
    )
    M_R2_RN2_r5 = M_R2_RN2._replace(
        description='Same as M_R2_RN2, but with different random seed',
        random_state=1095,
    )
    M_R2_RN2_r6 = M_R2_RN2._replace(
        description='Same as M_R2_RN2, but with different random seed',
        random_state=1096,
    )
    M_R2_RN2_r7 = M_R2_RN2._replace(
        description='Same as M_R2_RN2, but with different random seed',
        random_state=1097,
    )
    M_R2_RN2_r8 = M_R2_RN2._replace(
        description='Same as M_R2_RN2, but with different random seed',
        random_state=1098,
    )
    M_R2_RN2_r9 = M_R2_RN2._replace(
        description='Same as M_R2_RN2, but with different random seed',
        random_state=1099,
    )

    # RN3 Ensemble
    M_R1_FF_RN3_r1 = M_R1_FF_RN3._replace(
        description='Same as M_R1_FF_RN3, but with different random seed',
        random_state=1101,
    )
    M_R1_FF_RN3_r2 = M_R1_FF_RN3._replace(
        description='Same as M_R1_FF_RN3, but with different random seed',
        random_state=1102,
    )
    M_R1_FF_RN3_r3 = M_R1_FF_RN3._replace(
        description='Same as M_R1_FF_RN3, but with different random seed',
        random_state=1103,
    )
    M_R1_FF_RN3_r4 = M_R1_FF_RN3._replace(
        description='Same as M_R1_FF_RN3, but with different random seed',
        random_state=1104,
    )
    M_R1_FF_RN3_r5 = M_R1_FF_RN3._replace(
        description='Same as M_R1_FF_RN3, but with different random seed',
        random_state=1105,
    )
    M_R1_FF_RN3_r6 = M_R1_FF_RN3._replace(
        description='Same as M_R1_FF_RN3, but with different random seed',
        random_state=1106,
    )
    M_R1_FF_RN3_r7 = M_R1_FF_RN3._replace(
        description='Same as M_R1_FF_RN3, but with different random seed',
        random_state=1107,
    )
    M_R1_FF_RN3_r8 = M_R1_FF_RN3._replace(
        description='Same as M_R1_FF_RN3, but with different random seed',
        random_state=1108,
    )
    M_R1_FF_RN3_r9 = M_R1_FF_RN3._replace(
        description='Same as M_R1_FF_RN3, but with different random seed',
        random_state=1109,
    )

    # RN4 Ensemble
    M_R2_FF_RN4_r1 = M_R2_FF_RN4._replace(
        description='Same as M_R2_FF_RN4, but with different random seed',
        random_state=1111,
    )
    M_R2_FF_RN4_r2 = M_R2_FF_RN4._replace(
        description='Same as M_R2_FF_RN4, but with different random seed',
        random_state=1112,
    )
    M_R2_FF_RN4_r3 = M_R2_FF_RN4._replace(
        description='Same as M_R2_FF_RN4, but with different random seed',
        random_state=1113,
    )
    M_R2_FF_RN4_r4 = M_R2_FF_RN4._replace(
        description='Same as M_R2_FF_RN4, but with different random seed',
        random_state=1114,
    )
    M_R2_FF_RN4_r5 = M_R2_FF_RN4._replace(
        description='Same as M_R2_FF_RN4, but with different random seed',
        random_state=1115,
    )
    M_R2_FF_RN4_r6 = M_R2_FF_RN4._replace(
        description='Same as M_R2_FF_RN4, but with different random seed',
        random_state=1116,
    )
    M_R2_FF_RN4_r7 = M_R2_FF_RN4._replace(
        description='Same as M_R2_FF_RN4, but with different random seed',
        random_state=1117,
    )
    M_R2_FF_RN4_r8 = M_R2_FF_RN4._replace(
        description='Same as M_R2_FF_RN4, but with different random seed',
        random_state=1118,
    )
    M_R2_FF_RN4_r9 = M_R2_FF_RN4._replace(
        description='Same as M_R2_FF_RN4, but with different random seed',
        random_state=1119,
    )

    # Anders Björkelund CNN model
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
    M_R1_FF_AB1 = M_R1_AB1._replace(
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
    M_R2_FF_AB1 = M_R1_AB1._replace(
        description='Same AB model, 2 ECGs + flat-features.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_tnt_1', 'male', 'age', 'log_dt']
            },
        },
    )

    # All models again, but with random splits!
    RS_M_F1_LR2 = Experiment(
        description='Scikit-learn logistic regression model, mace vs '
                    'features from Forberg et al. Features are normalized '
                    'and then reduced in dimension by PCA. Random split.',
        model=LogisticRegression,
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'forberg': ['ecg_0', 'combine']
            },
            'cv_kwargs': {'test_size': 0}
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'forberg_features': {
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
        },
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'test_size': 1 / 2,
            'random_state': 9001
        },
        scoring=roc_auc_score,
    )
    RS_M_F2_LR2 = RS_M_F1_LR2._replace(
        description='Scikit-learn logistic regression model, mace vs '
                    'features from Forberg et al. Features are normalized '
                    'and then reduced in dimension by PCA. Random split.',
        extractor_kwargs={
            "features": {
                'forberg': ['ecg_0', 'diff', 'combine']
            },
            'cv_kwargs': {'test_size': 0}
        },
        pre_processor_kwargs={
            'forberg_features': {
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
    )
    RS_M_F1_FF_LR2 = RS_M_F1_LR2._replace(
        description='Scikit-learn logistic regression model, mace vs '
                    'features from Forberg et al. Features are normalized '
                    'and then reduced in dimension by PCA. Random split.',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male', 'log_tnt_1'],
                'forberg': ['ecg_0', 'combine']
            },
            'cv_kwargs': {'test_size': 0}
        },
        pre_processor_kwargs={
            'forberg_features': {
                'processor': 'Pipeline',
                'steps': [
                    ('scaler', StandardScaler, {}),
                    ('pca', PCA, {
                        'n_components': 2,
                        'whiten': False,
                        'random_state': 42
                    })
                ]
            },
            'flat_features': {
                'processor': StandardScaler,
            }
        },
    )
    RS_M_F2_FF_LR2 = RS_M_F1_LR2._replace(
        description='Scikit-learn logistic regression model, mace vs '
                    'features from Forberg et al. Features are normalized '
                    'and then reduced in dimension by PCA. I tried a bunch '
                    'of settings for dimension, ~150 gave the best AUC. '
                    'Random split.',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male', 'log_tnt_1'],
                'forberg': ['ecg_0', 'diff', 'combine']
            },
            'cv_kwargs': {'test_size': 0}
        },
        pre_processor_kwargs={
            'forberg_features': {
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
            'flat_features': {
                'processor': StandardScaler,
            }
        },
    )

    RS_M_F1_NN1 = Experiment(
        description='Best iteration (xp_184) from M_F1_NN_RS. Random split.',
        model=ffnn,
        model_kwargs={
            'ecg_ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.3],
                'batch_norms': [False],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.001],
                'bias_regularizers': [0.1]
            },
            'ecg_combiner': None,
            'ecg_comb_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.0],
                'batch_norms': [False],
                'activity_regularizers': [0.0001],
                'kernel_regularizers': [0.01],
                'bias_regularizers': [0.001]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': None,
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'forberg': ['ecg_0']
            },
            'cv_kwargs': {'test_size': 0}
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'forberg_ecg_0': {'processor': StandardScaler},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.0001,
            }
        },
        epochs=100,
        batch_size=64,
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'test_size': 1 / 2,
            'random_state': 9001
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc'],
    )
    RS_M_F1_NN1_r1 = RS_M_F1_NN1._replace(
        description='Same as RS_M_F1_NN1, but with a different random seed.',
        random_state=1001,
    )
    RS_M_F1_NN1_r2 = RS_M_F1_NN1._replace(
        description='Same as RS_M_F1_NN1, but with a different random seed.',
        random_state=1002,
    )
    RS_M_F1_NN1_r3 = RS_M_F1_NN1._replace(
        description='Same as RS_M_F1_NN1, but with a different random seed.',
        random_state=1003,
    )
    RS_M_F1_NN1_r4 = RS_M_F1_NN1._replace(
        description='Same as RS_M_F1_NN1, but with a different random seed.',
        random_state=1004,
    )
    RS_M_F1_NN1_r5 = RS_M_F1_NN1._replace(
        description='Same as RS_M_F1_NN1, but with a different random seed.',
        random_state=1005,
    )
    RS_M_F1_NN1_r6 = RS_M_F1_NN1._replace(
        description='Same as RS_M_F1_NN1, but with a different random seed.',
        random_state=1006,
    )
    RS_M_F1_NN1_r7 = RS_M_F1_NN1._replace(
        description='Same as RS_M_F1_NN1, but with a different random seed.',
        random_state=1007,
    )
    RS_M_F1_NN1_r8 = RS_M_F1_NN1._replace(
        description='Same as RS_M_F1_NN1, but with a different random seed.',
        random_state=1008,
    )
    RS_M_F1_NN1_r9 = RS_M_F1_NN1._replace(
        description='Same as RS_M_F1_NN1, but with a different random seed.',
        random_state=1009,
    )

    RS_M_F2_NN2 = Experiment(
        description='Best iteration (xp_324) from M_F2_NN_RS. Random Split',
        model=ffnn,
        model_kwargs={
            'ecg_ffnn_kwargs': {
                'sizes': [200],
                'dropouts': [0.3],
                'batch_norms': [False],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.0001],
                'bias_regularizers': [0.01]
            },
            'ecg_combiner': 'concatenate',
            'ecg_comb_ffnn_kwargs': {
                'sizes': [6],
                'dropouts': [0.0],
                'batch_norms': [False],
                'activity_regularizers': [0.001],
                'kernel_regularizers': [0.1],
                'bias_regularizers': [0.0]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': None,
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'forberg': ['ecg_0', 'ecg_1']
            },
            'cv_kwargs': {'test_size': 0},
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'forberg_ecg_0': {'processor': StandardScaler},
            'forberg_ecg_1': {'processor': StandardScaler},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.0001,
            }
        },
        epochs=100,
        batch_size=64,
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'test_size': 1 / 2,
            'random_state': 9001
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc'],
    )
    RS_M_F2_NN2_r1 = RS_M_F2_NN2._replace(
        description='Same as RS_M_F2_NN2, but with a different random seed.',
        random_state=1011,
    )
    RS_M_F2_NN2_r2 = RS_M_F2_NN2._replace(
        description='Same as RS_M_F2_NN2, but with a different random seed.',
        random_state=1012,
    )
    RS_M_F2_NN2_r3 = RS_M_F2_NN2._replace(
        description='Same as RS_M_F2_NN2, but with a different random seed.',
        random_state=1013,
    )
    RS_M_F2_NN2_r4 = RS_M_F2_NN2._replace(
        description='Same as RS_M_F2_NN2, but with a different random seed.',
        random_state=1014,
    )
    RS_M_F2_NN2_r5 = RS_M_F2_NN2._replace(
        description='Same as RS_M_F2_NN2, but with a different random seed.',
        random_state=1015,
    )
    RS_M_F2_NN2_r6 = RS_M_F2_NN2._replace(
        description='Same as RS_M_F2_NN2, but with a different random seed.',
        random_state=1016,
    )
    RS_M_F2_NN2_r7 = RS_M_F2_NN2._replace(
        description='Same as RS_M_F2_NN2, but with a different random seed.',
        random_state=1017,
    )
    RS_M_F2_NN2_r8 = RS_M_F2_NN2._replace(
        description='Same as RS_M_F2_NN2, but with a different random seed.',
        random_state=1018,
    )
    RS_M_F2_NN2_r9 = RS_M_F2_NN2._replace(
        description='Same as RS_M_F2_NN2, but with a different random seed.',
        random_state=1019,
    )

    RS_M_F1_FF_NN3 = Experiment(
        description='Best iteration (xp_353) from M_F1_FF_NN_RS. Random '
                    'split.',
        model=ffnn,
        model_kwargs={
            'ecg_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.2],
                'batch_norms': [True],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.1],
                'bias_regularizers': [0.001]
            },
            'ecg_combiner': None,
            'ecg_comb_ffnn_kwargs': {
                'sizes': [20],
                'dropouts': [0.3],
                'batch_norms': [False],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.001],
                'bias_regularizers': [0.0]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.0],
                'batch_norms': [True],
                'activity_regularizers': [0.0001],
                'kernel_regularizers': [0.1],
                'bias_regularizers': [0.01]
            },
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt'],
                'forberg': ['ecg_0']
            },
            'cv_kwargs': {'test_size': 0},
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'flat_features': {'processor': StandardScaler},
            'forberg_ecg_0': {'processor': StandardScaler},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.0003,
            }
        },
        epochs=100,
        batch_size=64,
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'test_size': 1 / 2,
            'random_state': 9001
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc'],
    )
    RS_M_F1_FF_NN3_r1 = RS_M_F1_FF_NN3._replace(
        description='Same as RS_M_F1_FF_NN3, but with a different random '
                    'seed.',
        random_state=1021,
    )
    RS_M_F1_FF_NN3_r2 = RS_M_F1_FF_NN3._replace(
        description='Same as RS_M_F1_FF_NN3, but with a different random '
                    'seed.',
        random_state=1022,
    )
    RS_M_F1_FF_NN3_r3 = RS_M_F1_FF_NN3._replace(
        description='Same as RS_M_F1_FF_NN3, but with a different random '
                    'seed.',
        random_state=1023,
    )
    RS_M_F1_FF_NN3_r4 = RS_M_F1_FF_NN3._replace(
        description='Same as RS_M_F1_FF_NN3, but with a different random '
                    'seed.',
        random_state=1024,
    )
    RS_M_F1_FF_NN3_r5 = RS_M_F1_FF_NN3._replace(
        description='Same as RS_M_F1_FF_NN3, but with a different random '
                    'seed.',
        random_state=1025,
    )
    RS_M_F1_FF_NN3_r6 = RS_M_F1_FF_NN3._replace(
        description='Same as RS_M_F1_FF_NN3, but with a different random '
                    'seed.',
        random_state=1026,
    )
    RS_M_F1_FF_NN3_r7 = RS_M_F1_FF_NN3._replace(
        description='Same as RS_M_F1_FF_NN3, but with a different random '
                    'seed.',
        random_state=1027,
    )
    RS_M_F1_FF_NN3_r8 = RS_M_F1_FF_NN3._replace(
        description='Same as RS_M_F1_FF_NN3, but with a different random '
                    'seed.',
        random_state=1028,
    )
    RS_M_F1_FF_NN3_r9 = RS_M_F1_FF_NN3._replace(
        description='Same as RS_M_F1_FF_NN3, but with a different random '
                    'seed.',
        random_state=1029,
    )

    RS_M_F2_FF_NN6 = Experiment(
        description='Third best iteration (xp_208) from M_F2_FF_NN_RS. '
                    'Random split.',
        model=ffnn,
        model_kwargs={
            'ecg_ffnn_kwargs': {
                'sizes': [200],
                'dropouts': [0.3],
                'batch_norms': [False],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.001],
                'bias_regularizers': [0.1]
            },
            'ecg_combiner': 'difference',
            'ecg_comb_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.4],
                'batch_norms': [True],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.1],
                'bias_regularizers': [0.1]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.1],
                'batch_norms': [True],
                'activity_regularizers': [0.0],
                'kernel_regularizers': [0.0],
                'bias_regularizers': [0.0001]
            },
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt'],
                'forberg': ['ecg_0', 'ecg_1']
            },
            'cv_kwargs': {'test_size': 0},
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'flat_features': {'processor': StandardScaler},
            'forberg_ecg_0': {'processor': StandardScaler},
            'forberg_ecg_1': {'processor': StandardScaler},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.001,
            }
        },
        epochs=100,
        batch_size=64,
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'test_size': 1 / 2,
            'random_state': 9001
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc'],
    )
    RS_M_F2_FF_NN6_r1 = RS_M_F2_FF_NN6._replace(
        description='Same as RS_M_F2_FF_NN6, but with a different random '
                    'seed.',
        random_state=1031,
    )
    RS_M_F2_FF_NN6_r2 = RS_M_F2_FF_NN6._replace(
        description='Same as RS_M_F2_FF_NN6, but with a different random '
                    'seed.',
        random_state=1032,
    )
    RS_M_F2_FF_NN6_r3 = RS_M_F2_FF_NN6._replace(
        description='Same as RS_M_F2_FF_NN6, but with a different random '
                    'seed.',
        random_state=1033,
    )
    RS_M_F2_FF_NN6_r4 = RS_M_F2_FF_NN6._replace(
        description='Same as RS_M_F2_FF_NN6, but with a different random '
                    'seed.',
        random_state=1034,
    )
    RS_M_F2_FF_NN6_r5 = RS_M_F2_FF_NN6._replace(
        description='Same as RS_M_F2_FF_NN6, but with a different random '
                    'seed.',
        random_state=1035,
    )
    RS_M_F2_FF_NN6_r6 = RS_M_F2_FF_NN6._replace(
        description='Same as RS_M_F2_FF_NN6, but with a different random '
                    'seed.',
        random_state=1036,
    )
    RS_M_F2_FF_NN6_r7 = RS_M_F2_FF_NN6._replace(
        description='Same as RS_M_F2_FF_NN6, but with a different random '
                    'seed.',
        random_state=1037,
    )
    RS_M_F2_FF_NN6_r8 = RS_M_F2_FF_NN6._replace(
        description='Same as RS_M_F2_FF_NN6, but with a different random '
                    'seed.',
        random_state=1038,
    )
    RS_M_F2_FF_NN6_r9 = RS_M_F2_FF_NN6._replace(
        description='Same as RS_M_F2_FF_NN6, but with a different random '
                    'seed.',
        random_state=1039,
    )

    RS_M_R1_CNN1 = Experiment(
        description='Uses xp_210 from M_R1_CNN_RS, which was the second '
                    'best in terms of AUC, but looked better than the best '
                    'when considering the overall learning trend. '
                    'Random split.',
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
            'cv_kwargs': {'test_size': 0},
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.0001}
        },
        epochs=100,
        batch_size=64,
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'test_size': 1 / 2,
            'random_state': 9001
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    RS_M_R1_CNN1_r1 = RS_M_R1_CNN1._replace(
        description='Same as RS_M_R1_CNN1, but with a different random seed.',
        random_state=1041,
    )
    RS_M_R1_CNN1_r2 = RS_M_R1_CNN1._replace(
        description='Same as RS_M_R1_CNN1, but with a different random seed.',
        random_state=1042,
    )
    RS_M_R1_CNN1_r3 = RS_M_R1_CNN1._replace(
        description='Same as RS_M_R1_CNN1, but with a different random seed.',
        random_state=1043,
    )
    RS_M_R1_CNN1_r4 = RS_M_R1_CNN1._replace(
        description='Same as RS_M_R1_CNN1, but with a different random seed.',
        random_state=1044,
    )
    RS_M_R1_CNN1_r5 = RS_M_R1_CNN1._replace(
        description='Same as RS_M_R1_CNN1, but with a different random seed.',
        random_state=1045,
    )
    RS_M_R1_CNN1_r6 = RS_M_R1_CNN1._replace(
        description='Same as RS_M_R1_CNN1, but with a different random seed.',
        random_state=1046,
    )
    RS_M_R1_CNN1_r7 = RS_M_R1_CNN1._replace(
        description='Same as RS_M_R1_CNN1, but with a different random seed.',
        random_state=1047,
    )
    RS_M_R1_CNN1_r8 = RS_M_R1_CNN1._replace(
        description='Same as RS_M_R1_CNN1, but with a different random seed.',
        random_state=1048,
    )
    RS_M_R1_CNN1_r9 = RS_M_R1_CNN1._replace(
        description='Same as RS_M_R1_CNN1, but with a different random seed.',
        random_state=1049,
    )

    RS_M_R2_CNN2 = Experiment(
        description='Uses xp_26 from M_R2_CNN_RS, which was best in terms '
                    'of AUC. Random split.',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'down_sample': True,
                'dropouts': [0.0, 0.2],
                'pool_size': 19,
                'filter_first': 32,
                'filter_last': 64,
                'kernel_first': 49,
                'kernel_last': 17,
                'batch_norms': [False, True],
                'weight_decays': [0.001, 0.1],
            },
            'ecg_ffnn_kwargs': None,
            'ecg_comb_ffnn_kwargs': {
                'sizes': [20],
                'dropouts': [0.2],
                'batch_norms': [False]
            },
            'ecg_combiner': 'concatenate',
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [20],
                'dropouts': [0.0],
                'batch_norms': [False]
            }
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
            },
            'cv_kwargs': {'test_size': 0},
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.00001}
        },
        epochs=100,
        batch_size=64,
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'test_size': 1 / 2,
            'random_state': 9001
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    RS_M_R2_CNN2_r1 = RS_M_R2_CNN2._replace(
        description='Same as RS_M_R2_CNN2, but with a different random seed.',
        random_state=1051,
    )
    RS_M_R2_CNN2_r2 = RS_M_R2_CNN2._replace(
        description='Same as RS_M_R2_CNN2, but with a different random seed.',
        random_state=1052,
    )
    RS_M_R2_CNN2_r3 = RS_M_R2_CNN2._replace(
        description='Same as RS_M_R2_CNN2, but with a different random seed.',
        random_state=1053,
    )
    RS_M_R2_CNN2_r4 = RS_M_R2_CNN2._replace(
        description='Same as RS_M_R2_CNN2, but with a different random seed.',
        random_state=1054,
    )
    RS_M_R2_CNN2_r5 = RS_M_R2_CNN2._replace(
        description='Same as RS_M_R2_CNN2, but with a different random seed.',
        random_state=1055,
    )
    RS_M_R2_CNN2_r6 = RS_M_R2_CNN2._replace(
        description='Same as RS_M_R2_CNN2, but with a different random seed.',
        random_state=1056,
    )
    RS_M_R2_CNN2_r7 = RS_M_R2_CNN2._replace(
        description='Same as RS_M_R2_CNN2, but with a different random seed.',
        random_state=1057,
    )
    RS_M_R2_CNN2_r8 = RS_M_R2_CNN2._replace(
        description='Same as RS_M_R2_CNN2, but with a different random seed.',
        random_state=1058,
    )
    RS_M_R2_CNN2_r9 = RS_M_R2_CNN2._replace(
        description='Same as RS_M_R2_CNN2, but with a different random seed.',
        random_state=1059,
    )

    RS_M_R1_FF_CNN3 = Experiment(
        description='Uses xp_379 from M_R1_FF_CNN_RS, which was the top '
                    'performing model found after 400 iterations of random '
                    'search. Random split.',
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
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
            'cv_kwargs': {'test_size': 0},
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.001}
        },
        epochs=100,
        batch_size=64,
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'test_size': 1 / 2,
            'random_state': 9001
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    RS_M_R1_FF_CNN3_r1 = RS_M_R1_FF_CNN3._replace(
        description='Same as RS_M_R1_FF_CNN3, but with a different random '
                    'seed.',
        random_state=1061,
    )
    RS_M_R1_FF_CNN3_r2 = RS_M_R1_FF_CNN3._replace(
        description='Same as RS_M_R1_FF_CNN3, but with a different random '
                    'seed.',
        random_state=1062,
    )
    RS_M_R1_FF_CNN3_r3 = RS_M_R1_FF_CNN3._replace(
        description='Same as RS_M_R1_FF_CNN3, but with a different random '
                    'seed.',
        random_state=1063,
    )
    RS_M_R1_FF_CNN3_r4 = RS_M_R1_FF_CNN3._replace(
        description='Same as RS_M_R1_FF_CNN3, but with a different random '
                    'seed.',
        random_state=1064,
    )
    RS_M_R1_FF_CNN3_r5 = RS_M_R1_FF_CNN3._replace(
        description='Same as RS_M_R1_FF_CNN3, but with a different random '
                    'seed.',
        random_state=1065,
    )
    RS_M_R1_FF_CNN3_r6 = RS_M_R1_FF_CNN3._replace(
        description='Same as RS_M_R1_FF_CNN3, but with a different random '
                    'seed.',
        random_state=1066,
    )
    RS_M_R1_FF_CNN3_r7 = RS_M_R1_FF_CNN3._replace(
        description='Same as RS_M_R1_FF_CNN3, but with a different random '
                    'seed.',
        random_state=1067,
    )
    RS_M_R1_FF_CNN3_r8 = RS_M_R1_FF_CNN3._replace(
        description='Same as RS_M_R1_FF_CNN3, but with a different random '
                    'seed.',
        random_state=1068,
    )
    RS_M_R1_FF_CNN3_r9 = RS_M_R1_FF_CNN3._replace(
        description='Same as RS_M_R1_FF_CNN3, but with a different random '
                    'seed.',
        random_state=1069,
    )

    RS_M_R2_FF_CNN4 = Experiment(
        description='Uses xp_379 from M_R1_FF_CNN_RS, which was the top '
                    'performing model found after 400 iterations of random '
                    'search. Random split.',
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
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
            'cv_kwargs': {'test_size': 0},
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.001}
        },
        epochs=100,
        batch_size=64,
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'test_size': 1 / 2,
            'random_state': 9001
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    RS_M_R2_FF_CNN4_r1 = RS_M_R2_FF_CNN4._replace(
        description='Same as RS_M_R2_FF_CNN4, but with a different random '
                    'seed.',
        random_state=1071,
    )
    RS_M_R2_FF_CNN4_r2 = RS_M_R2_FF_CNN4._replace(
        description='Same as RS_M_R2_FF_CNN4, but with a different random '
                    'seed.',
        random_state=1072,
    )
    RS_M_R2_FF_CNN4_r3 = RS_M_R2_FF_CNN4._replace(
        description='Same as RS_M_R2_FF_CNN4, but with a different random '
                    'seed.',
        random_state=1073,
    )
    RS_M_R2_FF_CNN4_r4 = RS_M_R2_FF_CNN4._replace(
        description='Same as RS_M_R2_FF_CNN4, but with a different random '
                    'seed.',
        random_state=1074,
    )
    RS_M_R2_FF_CNN4_r5 = RS_M_R2_FF_CNN4._replace(
        description='Same as RS_M_R2_FF_CNN4, but with a different random '
                    'seed.',
        random_state=1075,
    )
    RS_M_R2_FF_CNN4_r6 = RS_M_R2_FF_CNN4._replace(
        description='Same as RS_M_R2_FF_CNN4, but with a different random '
                    'seed.',
        random_state=1076,
    )
    RS_M_R2_FF_CNN4_r7 = RS_M_R2_FF_CNN4._replace(
        description='Same as RS_M_R2_FF_CNN4, but with a different random '
                    'seed.',
        random_state=1077,
    )
    RS_M_R2_FF_CNN4_r8 = RS_M_R2_FF_CNN4._replace(
        description='Same as RS_M_R2_FF_CNN4, but with a different random '
                    'seed.',
        random_state=1078,
    )
    RS_M_R2_FF_CNN4_r9 = RS_M_R2_FF_CNN4._replace(
        description='Same as RS_M_R2_FF_CNN4, but with a different random '
                    'seed.',
        random_state=1079,
    )

    RS_M_R1_RN1 = Experiment(
        description='xp_141 from R1_RN_RS. Random split.',
        model=pretrained_resnet,
        model_kwargs={
            'freeze_resnet': False,
            'ecg_ffnn_kwargs': {
                'sizes': [200, 100],
                'dropouts': [0.0, 0.1],
                'batch_norms': [False, True],
                'activity_regularizers': [0.0, 0.0001],
                'kernel_regularizers': [0.01, 0.0],
                'bias_regularizers': [0.0001, 0.1]
            },
            'ecg_combiner': 'difference',
            'ecg_comb_ffnn_kwargs': {
                'sizes': [6],
                'dropouts': [0.0],
                'batch_norms': [False],
                'activity_regularizers': [0.0001],
                'kernel_regularizers': [0.1],
                'bias_regularizers': [0.001]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': None,
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            },
            'cv_kwargs': {'test_size': 0},
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.0003}
        },
        epochs=50,
        batch_size=32,
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'test_size': 1 / 2,
            'random_state': 9001
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    RS_M_R1_RN1_r1 = RS_M_R1_RN1._replace(
        description='Same as RS_M_R1_RN1, but with a different random seed.',
        random_state=1081,
    )
    RS_M_R1_RN1_r2 = RS_M_R1_RN1._replace(
        description='Same as RS_M_R1_RN1, but with a different random seed.',
        random_state=1082,
    )
    RS_M_R1_RN1_r3 = RS_M_R1_RN1._replace(
        description='Same as RS_M_R1_RN1, but with a different random seed.',
        random_state=1083,
    )
    RS_M_R1_RN1_r4 = RS_M_R1_RN1._replace(
        description='Same as RS_M_R1_RN1, but with a different random seed.',
        random_state=1084,
    )
    RS_M_R1_RN1_r5 = RS_M_R1_RN1._replace(
        description='Same as RS_M_R1_RN1, but with a different random seed.',
        random_state=1085,
    )
    RS_M_R1_RN1_r6 = RS_M_R1_RN1._replace(
        description='Same as RS_M_R1_RN1, but with a different random seed.',
        random_state=1086,
    )
    RS_M_R1_RN1_r7 = RS_M_R1_RN1._replace(
        description='Same as RS_M_R1_RN1, but with a different random seed.',
        random_state=1087,
    )
    RS_M_R1_RN1_r8 = RS_M_R1_RN1._replace(
        description='Same as RS_M_R1_RN1, but with a different random seed.',
        random_state=1088,
    )
    RS_M_R1_RN1_r9 = RS_M_R1_RN1._replace(
        description='Same as RS_M_R1_RN1, but with a different random seed.',
        random_state=1089,
    )

    RS_M_R2_RN2 = Experiment(
        description='xp_133 from R1_RN_RS, 2nd best model. Random split.',
        model=pretrained_resnet,
        model_kwargs={
            'freeze_resnet': False,
            'ecg_ffnn_kwargs': {
                'sizes': [200, 50],
                'dropouts': [0.2, 0.0],
                'batch_norms': [False, False],
                'activity_regularizers': [0.001, 0.00001],
                'kernel_regularizers': [0.01, 0.01],
                'bias_regularizers': [0.01, 0.001]
            },
            'ecg_combiner': 'difference',
            'ecg_comb_ffnn_kwargs': {
                'sizes': [20],
                'dropouts': [0.3],
                'batch_norms': [False],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.01],
                'bias_regularizers': [0.01]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': None,
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            },
            'cv_kwargs': {'test_size': 0},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [5490],
                        'values': [0.001, 0.00001]
                    }
                }
            }
        },
        epochs=50,
        batch_size=32,
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'test_size': 1 / 2,
            'random_state': 9001
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    RS_M_R2_RN2_r1 = RS_M_R2_RN2._replace(
        description='Same as RS_M_R2_RN2, but with a different random seed.',
        random_state=1091,
    )
    RS_M_R2_RN2_r2 = RS_M_R2_RN2._replace(
        description='Same as RS_M_R2_RN2, but with a different random seed.',
        random_state=1092,
    )
    RS_M_R2_RN2_r3 = RS_M_R2_RN2._replace(
        description='Same as RS_M_R2_RN2, but with a different random seed.',
        random_state=1093,
    )
    RS_M_R2_RN2_r4 = RS_M_R2_RN2._replace(
        description='Same as RS_M_R2_RN2, but with a different random seed.',
        random_state=1094,
    )
    RS_M_R2_RN2_r5 = RS_M_R2_RN2._replace(
        description='Same as RS_M_R2_RN2, but with a different random seed.',
        random_state=1095,
    )
    RS_M_R2_RN2_r6 = RS_M_R2_RN2._replace(
        description='Same as RS_M_R2_RN2, but with a different random seed.',
        random_state=1096,
    )
    RS_M_R2_RN2_r7 = RS_M_R2_RN2._replace(
        description='Same as RS_M_R2_RN2, but with a different random seed.',
        random_state=1097,
    )
    RS_M_R2_RN2_r8 = RS_M_R2_RN2._replace(
        description='Same as RS_M_R2_RN2, but with a different random seed.',
        random_state=1098,
    )
    RS_M_R2_RN2_r9 = RS_M_R2_RN2._replace(
        description='Same as RS_M_R2_RN2, but with a different random seed.',
        random_state=1099,
    )

    RS_M_R1_FF_RN3 = Experiment(
        description='xp_166 from R1_FF_RN_RS, 2nd best model. Random split.',
        model=pretrained_resnet,
        model_kwargs={
            'freeze_resnet': False,
            'ecg_ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.5],
                'batch_norms': [False],
                'activity_regularizers': [0.00001],
                'kernel_regularizers': [0.01],
                'bias_regularizers': [0.0],
            },
            'ecg_combiner': 'concatenate',
            'ecg_comb_ffnn_kwargs': {
                'sizes': [6],
                'dropouts': [0.0],
                'batch_norms': [True],
                'activity_regularizers': [0.0001],
                'kernel_regularizers': [0.01],
                'bias_regularizers': [0.1]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.2],
                'batch_norms': [True],
                'activity_regularizers': [0.0001],
                'kernel_regularizers': [0.0],
                'bias_regularizers': [0.0001],
            },
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            },
            'cv_kwargs': {'test_size': 0},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [7320],
                        'values': [0.0003, 0.00003]
                    }
                },
            }
        },
        epochs=50,
        batch_size=32,
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'test_size': 1 / 2,
            'random_state': 9001
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    RS_M_R1_FF_RN3_r1 = RS_M_R1_FF_RN3._replace(
        description='Same as RS_M_R1_FF_RN3, but with a different random '
                    'seed.',
        random_state=1101,
    )
    RS_M_R1_FF_RN3_r2 = RS_M_R1_FF_RN3._replace(
        description='Same as RS_M_R1_FF_RN3, but with a different random '
                    'seed.',
        random_state=1102,
    )
    RS_M_R1_FF_RN3_r3 = RS_M_R1_FF_RN3._replace(
        description='Same as RS_M_R1_FF_RN3, but with a different random '
                    'seed.',
        random_state=1103,
    )
    RS_M_R1_FF_RN3_r4 = RS_M_R1_FF_RN3._replace(
        description='Same as RS_M_R1_FF_RN3, but with a different random '
                    'seed.',
        random_state=1104,
    )
    RS_M_R1_FF_RN3_r5 = RS_M_R1_FF_RN3._replace(
        description='Same as RS_M_R1_FF_RN3, but with a different random '
                    'seed.',
        random_state=1105,
    )
    RS_M_R1_FF_RN3_r6 = RS_M_R1_FF_RN3._replace(
        description='Same as RS_M_R1_FF_RN3, but with a different random '
                    'seed.',
        random_state=1106,
    )
    RS_M_R1_FF_RN3_r7 = RS_M_R1_FF_RN3._replace(
        description='Same as RS_M_R1_FF_RN3, but with a different random '
                    'seed.',
        random_state=1107,
    )
    RS_M_R1_FF_RN3_r8 = RS_M_R1_FF_RN3._replace(
        description='Same as RS_M_R1_FF_RN3, but with a different random '
                    'seed.',
        random_state=1108,
    )
    RS_M_R1_FF_RN3_r9 = RS_M_R1_FF_RN3._replace(
        description='Same as RS_M_R1_FF_RN3, but with a different random '
                    'seed.',
        random_state=1109,
    )

    RS_M_R2_FF_RN4 = Experiment(
        description='xp_24 from R2_FF_RN_RS. Random split.',
        model=pretrained_resnet,
        model_kwargs={
            'freeze_resnet': False,
            'ecg_ffnn_kwargs': {
                'sizes': [25],
                'dropouts': [0.2],
                'batch_norms': [False],
                'activity_regularizers': [0.01],
                'kernel_regularizers': [0.001],
                'bias_regularizers': [0.001]
            },
            'ecg_combiner': 'difference',
            'ecg_comb_ffnn_kwargs': {
                'sizes': [6],
                'dropouts': [0.0],
                'batch_norms': [False],
                'activity_regularizers': [0.0001],
                'kernel_regularizers': [0.0001],
                'bias_regularizers': [0.0001]
            },
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.2],
                'batch_norms': [True],
                'activity_regularizers': [0.0],
                'kernel_regularizers': [0.0001],
                'bias_regularizers': [0.01],
            },
        },
        extractor=EscTrop,
        extractor_kwargs={
            'features': {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            },
            'cv_kwargs': {'test_size': 0},
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [10065],
                        'values': [0.003, 0.00001]
                    }
                },
            }
        },
        epochs=50,
        batch_size=32,
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'test_size': 1 / 2,
            'random_state': 9001
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
    )
    RS_M_R2_FF_RN4_r1 = RS_M_R2_FF_RN4._replace(
        description='Same as RS_M_R2_FF_RN4, but with a different random '
                    'seed.',
        random_state=1111,
    )
    RS_M_R2_FF_RN4_r2 = RS_M_R2_FF_RN4._replace(
        description='Same as RS_M_R2_FF_RN4, but with a different random '
                    'seed.',
        random_state=1112,
    )
    RS_M_R2_FF_RN4_r3 = RS_M_R2_FF_RN4._replace(
        description='Same as RS_M_R2_FF_RN4, but with a different random '
                    'seed.',
        random_state=1113,
    )
    RS_M_R2_FF_RN4_r4 = RS_M_R2_FF_RN4._replace(
        description='Same as RS_M_R2_FF_RN4, but with a different random '
                    'seed.',
        random_state=1114,
    )
    RS_M_R2_FF_RN4_r5 = RS_M_R2_FF_RN4._replace(
        description='Same as RS_M_R2_FF_RN4, but with a different random '
                    'seed.',
        random_state=1115,
    )
    RS_M_R2_FF_RN4_r6 = RS_M_R2_FF_RN4._replace(
        description='Same as RS_M_R2_FF_RN4, but with a different random '
                    'seed.',
        random_state=1116,
    )
    RS_M_R2_FF_RN4_r7 = RS_M_R2_FF_RN4._replace(
        description='Same as RS_M_R2_FF_RN4, but with a different random '
                    'seed.',
        random_state=1117,
    )
    RS_M_R2_FF_RN4_r8 = RS_M_R2_FF_RN4._replace(
        description='Same as RS_M_R2_FF_RN4, but with a different random '
                    'seed.',
        random_state=1118,
    )
    RS_M_R2_FF_RN4_r9 = RS_M_R2_FF_RN4._replace(
        description='Same as RS_M_R2_FF_RN4, but with a different random '
                    'seed.',
        random_state=1119,
    )


class HyperSearch(HyperExperiment, Enum):
    AMI_R1_CNN_RS = HyperExperiment(
        template=Experiment(
            description="Try to find good settings for predicting AMI using "
                        "a single raw ECG.",
            model=ecg_cnn,
            model_kwargs={
                'cnn_kwargs': hp.Choice([
                    {
                        'downsample': True,
                        'num_layers': num_layers,
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers),
                        'filter_first': hp.Int(8, 64, step=4),
                        'filter_last': hp.Int(8, 64, step=4),
                        'kernel_first': hp.Int(5, 65, step=4),
                        'kernel_last': hp.Int(5, 65, step=4),
                        'batch_norms': hp.Choices([True, False], k=num_layers),
                        'weight_decays': hp.Choices(
                            [1e-1, 1e-2, 1e-3, 0.0],
                            k=num_layers),
                        'ffnn_kwargs': hp.Choice([
                            None,
                            {
                                'sizes': hp.Choices([10, 50, 100], k=1),
                                'dropouts': hp.Choices(
                                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                                'batch_norms': [False]
                            }
                        ]),
                    } for num_layers in [2, 3, 4]
                ]),
                'ecg_ffnn_kwargs': None,
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.Choices([10, 20, 50, 100], k=1),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                        'batch_norms': [False]
                    }
                ]),
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0']
                },
                'labels': {'target': 'ami30'}
            },
            class_weight=hp.Choice([None, {0: 1, 1: 10.7}]),
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': {
                        'scheduler': PiecewiseConstantDecay,
                        'scheduler_kwargs': {
                            'boundaries': [153 * 50],
                            'values': hp.Choice([
                                [1e-2, 1e-3],
                                [1e-3, 1e-4],
                                [1e-4, 1e-5],
                                [1e-5, 1e-6]
                            ]),
                        }
                    },
                }
            },
            epochs=100,
            batch_size=64,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1/3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=42,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 500
        },
    )
    AMI_R1_CNN_HB = HyperExperiment(
        template=Experiment(
            description="Try to find good settings for predicting AMI using "
                        "a single raw ECG. Hyperband searcher.",
            model=ecg_cnn,
            model_kwargs={
                'cnn_kwargs': hp.Choice([
                    {
                        'downsample': True,
                        'num_layers': num_layers,
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers),
                        'filter_first': hp.Int(8, 64, step=4),
                        'filter_last': hp.Int(8, 64, step=4),
                        'kernel_first': hp.Int(5, 65, step=4),
                        'kernel_last': hp.Int(5, 65, step=4),
                        'batch_norms': hp.Choices([True, False], k=num_layers),
                        'weight_decays': hp.Choices(
                            [1e-1, 1e-2, 1e-3, 0.0],
                            k=num_layers),
                        'ffnn_kwargs': hp.Choice([
                            None,
                            {
                                'sizes': hp.Choices([10, 50, 100], k=1),
                                'dropouts': hp.Choices(
                                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                                'batch_norms': [False]
                            }
                        ]),
                    } for num_layers in [2, 3, 4]
                ]),
                'ecg_ffnn_kwargs': None,
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.Choices([10, 20, 50, 100], k=1),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                        'batch_norms': [False]
                    }
                ]),
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0']
                },
                'labels': {'target': 'ami30'}
            },
            class_weight={0: 1, 1: 10.7},
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
                }
            },
            epochs=100,
            batch_size=64,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1/3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=42,
        strategy=Hyperband,
        strategy_kwargs={
            'iterations': 20,
            'maximum_resource': 40,
            'resource_unit': 5
        }
    )

    M_R1_FF_CNN_RS = HyperExperiment(
        template=Experiment(
            description="Try to find good settings for predicting MACE using "
                        "1 raw ECG and flat-features.",
            model=ecg_cnn,
            model_kwargs={
                'cnn_kwargs': hp.Choice([
                    {
                        'downsample': True,
                        'num_layers': num_layers,
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers),
                        'pool_size': hp.Choice(pool_sizes),
                        'filter_first': hp.Int(8, 64, step=4),
                        'filter_last': hp.Int(8, 64, step=4),
                        'kernel_first': hp.Int(5, 65, step=4),
                        'kernel_last': hp.Int(5, 65, step=4),
                        'batch_norms': hp.Choices([True, False], k=num_layers),
                        'weight_decays': hp.Choices(
                            [1e-1, 1e-2, 1e-3, 0.0],
                            k=num_layers),
                        'ffnn_kwargs': hp.Choice([
                            None,
                            {
                                'sizes': hp.Choices([10, 50, 100], k=1),
                                'dropouts': hp.Choices(
                                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                                'batch_norms': [False]
                            }
                        ]),
                    } for num_layers, pool_sizes in zip(
                        [2, 3, 4],
                        [range(7, 31), range(4, 11), range(3, 7)]
                    )
                ]),
                'ecg_ffnn_kwargs': None,
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.Choices([10, 20, 50, 100], k=1),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                        'batch_norms': [False]
                    }
                ]),
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0'],
                    'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
                },
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
                }
            },
            epochs=100,
            batch_size=64,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1/3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=42,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )
    M_R2_FF_CNN_RS = HyperExperiment(
        template=Experiment(
            description="Try to find good settings for predicting MACE using "
                        "2 raw ECGs, with flat-features.",
            model=ecg_cnn,
            model_kwargs={
                'cnn_kwargs': hp.Choice([
                    {
                        'downsample': True,
                        'num_layers': num_layers,
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers),
                        'pool_size': hp.Choice(pool_sizes),
                        'filter_first': hp.Int(8, 64, step=4),
                        'filter_last': hp.Int(8, 64, step=4),
                        'kernel_first': hp.Int(5, 65, step=4),
                        'kernel_last': hp.Int(5, 65, step=4),
                        'batch_norms': hp.Choices([True, False], k=num_layers),
                        'weight_decays': hp.Choices(
                            [1e-1, 1e-2, 1e-3, 0.0],
                            k=num_layers),
                        'ffnn_kwargs': hp.Choice([
                            None,
                            {
                                'sizes': hp.Choices([10, 50, 100], k=1),
                                'dropouts': hp.Choices(
                                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                                'batch_norms': [False]
                            }
                        ]),
                    } for num_layers, pool_sizes in zip(
                        [2, 3, 4],
                        [range(7, 31), range(4, 11), range(3, 7)]
                    )
                ]),
                'cnn_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_ffnn_kwargs': hp.Choice([
                    None,
                    {
                        'sizes': hp.Choices([10, 20], k=1),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                        'batch_norms': [False]
                    }
                ]),
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.Choices([10, 20, 50, 100], k=1),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                        'batch_norms': [False]
                    }
                ]),
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0', 'ecg_1'],
                    'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
                },
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
                }
            },
            epochs=100,
            batch_size=64,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1/3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=43,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )
    M_R1_CNN_RS = HyperExperiment(
        template=Experiment(
            description="Try to find good settings for predicting MACE using "
                        "1 raw ECG, without flat-features.",
            model=ecg_cnn,
            model_kwargs={
                'cnn_kwargs': hp.Choice([
                    {
                        'down_sample': True,
                        'num_layers': num_layers,
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers),
                        'pool_size': hp.Choice(pool_sizes),
                        'filter_first': hp.Int(8, 64, step=4),
                        'filter_last': hp.Int(8, 64, step=4),
                        'kernel_first': hp.Int(5, 65, step=4),
                        'kernel_last': hp.Int(5, 65, step=4),
                        'batch_norms': hp.Choices([True, False], k=num_layers),
                        'weight_decays': hp.Choices(
                            [1e-1, 1e-2, 1e-3, 0.0],
                            k=num_layers),
                    } for num_layers, pool_sizes in zip(
                        [2, 3, 4],
                        [range(7, 31), range(4, 11), range(3, 7)]
                    )
                ]),
                'ecg_ffnn_kwargs': hp.Choice([
                    None,
                    {
                        'sizes': hp.Choices([10, 50, 100], k=1),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                        'batch_norms': [False]
                    }
                ]),
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.Choices([10, 20, 50, 100], k=1),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                        'batch_norms': [False]
                    }
                ]),
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0']
                },
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
                }
            },
            epochs=100,
            batch_size=64,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1/3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=44,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )
    M_R2_CNN_RS = HyperExperiment(
        template=Experiment(
            description="Try to find good settings for predicting MACE using "
                        "2 raw ECGs, without flat-features.",
            model=ecg_cnn,
            model_kwargs={
                'cnn_kwargs': hp.Choice([
                    {
                        'downsample': True,
                        'num_layers': num_layers,
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers),
                        'pool_size': hp.Choice(pool_sizes),
                        'filter_first': hp.Int(8, 64, step=4),
                        'filter_last': hp.Int(8, 64, step=4),
                        'kernel_first': hp.Int(5, 65, step=4),
                        'kernel_last': hp.Int(5, 65, step=4),
                        'batch_norms': hp.Choices([True, False], k=num_layers),
                        'weight_decays': hp.Choices(
                            [1e-1, 1e-2, 1e-3, 0.0],
                            k=num_layers),
                        'ffnn_kwargs': hp.Choice([
                            None,
                            {
                                'sizes': hp.Choices([10, 50, 100], k=1),
                                'dropouts': hp.Choices(
                                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                                'batch_norms': [False]
                            }
                        ]),
                    } for num_layers, pool_sizes in zip(
                        [2, 3, 4],
                        [range(7, 31), range(4, 11), range(3, 7)]
                    )
                ]),
                'cnn_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_ffnn_kwargs': hp.Choice([
                    None,
                    {
                        'sizes': hp.Choices([10, 20], k=1),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                        'batch_norms': [False]
                    }
                ]),
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.Choices([10, 20, 50, 100], k=1),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                        'batch_norms': [False]
                    }
                ]),
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0', 'ecg_1']
                },
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
                }
            },
            epochs=100,
            batch_size=64,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1/3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=44,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )

    M_R1_RN_RS = HyperExperiment(
        template=Experiment(
            description="Random search for pretrained resnet using 1 ECG. "
                        "Search space is over ffnn parameters "
                        "following the flatten layer of the resnet. ",
            model=pretrained_resnet,
            model_kwargs={
                'freeze_resnet': False,
                'ecg_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [25, 50, 100, 200],
                            k=num_layers,
                            ascending=False,
                        ),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'batch_norms': hp.Choices(
                            [True, False],
                            k=num_layers
                        ),
                        'activity_regularizers': hp.Choices(
                            [0.01, 0.001, 0.0001, 0.00001, 0.0],
                            k=num_layers
                        ),
                        'kernel_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        ),
                        'bias_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2]
                ]),
                'ecg_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_comb_ffnn_kwargs': {
                    'sizes': hp.Choices([6, 10, 20], k=1),
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': None,
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0'],
                },
                'processing': {
                    'scale': 1000,
                    'ribeiro': True
                }
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': {
                        'scheduler': PiecewiseConstantDecay,
                        'scheduler_kwargs': {
                            'boundaries': [hp.Int(1525, 7625, step=305)],
                            'values': hp.SortedChoices(
                                [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
                                k=2, ascending=False
                            )
                        }
                    },
                }
            },
            epochs=50,
            batch_size=32,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1 / 3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=46,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 200
        },
    )
    M_R2_RN_RS = HyperExperiment(
        template=Experiment(
            description="Random search for pretrained resnet using 2 ECGs. "
                        "Search space is over ffnn parameters "
                        "following the flatten layer of the resnet. ",
            model=pretrained_resnet,
            model_kwargs={
                'freeze_resnet': False,
                'ecg_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [25, 50, 100, 200],
                            k=num_layers,
                            ascending=False,
                        ),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'batch_norms': hp.Choices(
                            [True, False],
                            k=num_layers
                        ),
                        'activity_regularizers': hp.Choices(
                            [0.01, 0.001, 0.0001, 0.00001, 0.0],
                            k=num_layers
                        ),
                        'kernel_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        ),
                        'bias_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2]
                ]),
                'ecg_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_comb_ffnn_kwargs': {
                    'sizes': hp.Choices([6, 10, 20], k=1),
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': None,
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0', 'ecg_1'],
                },
                'processing': {
                    'scale': 1000,
                    'ribeiro': True
                }
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': {
                        'scheduler': PiecewiseConstantDecay,
                        'scheduler_kwargs': {
                            'boundaries': [hp.Int(1525, 7625, step=305)],
                            'values': hp.SortedChoices(
                                [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
                                k=2, ascending=False
                            )
                        }
                    },
                }
            },
            epochs=50,
            batch_size=32,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1 / 3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=47,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 200
        },
    )
    M_R1_FF_RN_RS = HyperExperiment(
        template=Experiment(
            description="Random search for pretrained resnet using 1 ECG + "
                        "flat-features. Search space is over ffnn parameters "
                        "following the flatten layer of the resnet. ",
            model=pretrained_resnet,
            model_kwargs={
                'freeze_resnet': False,
                'ecg_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [25, 50, 100, 200],
                            k=num_layers,
                            ascending=False,
                        ),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'batch_norms': hp.Choices(
                            [True, False],
                            k=num_layers
                        ),
                        'activity_regularizers': hp.Choices(
                            [0.01, 0.001, 0.0001, 0.00001, 0.0],
                            k=num_layers
                        ),
                        'kernel_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        ),
                        'bias_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2]
                ]),
                'ecg_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_comb_ffnn_kwargs': {
                    'sizes': hp.Choices([6, 10, 20], k=1),
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': {
                    'sizes': [10],
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        k=1
                    ),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0'],
                    'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
                },
                'processing': {
                    'scale': 1000,
                    'ribeiro': True
                }
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': {
                        'scheduler': PiecewiseConstantDecay,
                        'scheduler_kwargs': {
                            'boundaries': [hp.Int(1525, 7625, step=305)],
                            'values': hp.SortedChoices(
                                [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
                                k=2, ascending=False
                            )
                        }
                    },
                }
            },
            epochs=50,
            batch_size=32,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1 / 3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=48,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 200
        },
    )
    M_R2_FF_RN_RS = HyperExperiment(
        template=Experiment(
            description="Random search for pretrained resnet using 2 ECGs + "
                        "flat-features. Search space is over ffnn parameters "
                        "following the flatten layer of the resnet. ",
            model=pretrained_resnet,
            model_kwargs={
                'freeze_resnet': False,
                'ecg_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [25, 50, 100, 200],
                            k=num_layers,
                            ascending=False,
                        ),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'batch_norms': hp.Choices(
                            [True, False],
                            k=num_layers
                        ),
                        'activity_regularizers': hp.Choices(
                            [0.01, 0.001, 0.0001, 0.00001, 0.0],
                            k=num_layers
                        ),
                        'kernel_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        ),
                        'bias_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2]
                ]),
                'ecg_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_comb_ffnn_kwargs': {
                    'sizes': hp.Choices([6, 10, 20], k=1),
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], k=1),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': {
                    'sizes': [10],
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        k=1
                    ),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'ecg_mode': 'raw',
                    'ecgs': ['ecg_0', 'ecg_1'],
                    'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt']
                },
                'processing': {
                    'scale': 1000,
                    'ribeiro': True
                }
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': {
                        'scheduler': PiecewiseConstantDecay,
                        'scheduler_kwargs': {
                            'boundaries': [hp.Int(1525, 7625, step=305)],
                            'values': hp.SortedChoices(
                                [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
                                k=2, ascending=False
                            )
                        }
                    },
                }
            },
            epochs=50,
            batch_size=32,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1 / 3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=49,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 200
        },
    )

    M_F1_NN_RS = HyperExperiment(
        template=Experiment(
            description="Random search over simple feed-forward neural "
                        "networks, using forberg features from 1 ECG",
            model=ffnn,
            model_kwargs={
                'ecg_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [25, 50, 100, 200],
                            k=num_layers,
                            ascending=False,
                        ),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'batch_norms': hp.Choices(
                            [True, False],
                            k=num_layers
                        ),
                        'activity_regularizers': hp.Choices(
                            [0.01, 0.001, 0.0001, 0.00001, 0.0],
                            k=num_layers
                        ),
                        'kernel_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        ),
                        'bias_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2]
                ]),
                'ecg_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_comb_ffnn_kwargs': {
                    'sizes': hp.Choices([6, 10, 20], k=1),
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        k=1
                    ),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': None
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'forberg': ['ecg_0']
                },
            },
            pre_processor=sklearn_process,
            pre_processor_kwargs={
                'forberg_ecg_0': {'processor': StandardScaler},
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
                }
            },
            epochs=100,
            batch_size=64,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1 / 3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=42,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )
    M_F2_NN_RS = HyperExperiment(
        template=Experiment(
            description="Random search over simple feed-forward neural "
                        "networks, using forberg features from 2 ECGs.",
            model=ffnn,
            model_kwargs={
                'ecg_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [25, 50, 100, 200],
                            k=num_layers,
                            ascending=False,
                        ),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'batch_norms': hp.Choices(
                            [True, False],
                            k=num_layers
                        ),
                        'activity_regularizers': hp.Choices(
                            [0.01, 0.001, 0.0001, 0.00001, 0.0],
                            k=num_layers
                        ),
                        'kernel_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        ),
                        'bias_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2]
                ]),
                'ecg_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_comb_ffnn_kwargs': {
                    'sizes': hp.Choices([6, 10, 20], k=1),
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        k=1
                    ),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': None,
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'forberg': ['ecg_0', 'ecg_1']
                },
            },
            pre_processor=sklearn_process,
            pre_processor_kwargs={
                'forberg_ecg_0': {'processor': StandardScaler},
                'forberg_ecg_1': {'processor': StandardScaler},
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
                }
            },
            epochs=100,
            batch_size=64,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1 / 3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=43,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )
    M_F1_FF_NN_RS = HyperExperiment(
        template=Experiment(
            description="Random search over simple feed-forward neural "
                        "networks, using forberg features from 1 ECG, and "
                        "flat-features. ",
            model=ffnn,
            model_kwargs={
                'ecg_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [25, 50, 100, 200],
                            k=num_layers,
                            ascending=False,
                        ),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'batch_norms': hp.Choices(
                            [True, False],
                            k=num_layers
                        ),
                        'activity_regularizers': hp.Choices(
                            [0.01, 0.001, 0.0001, 0.00001, 0.0],
                            k=num_layers
                        ),
                        'kernel_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        ),
                        'bias_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2]
                ]),
                'ecg_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_comb_ffnn_kwargs': {
                    'sizes': hp.Choices([6, 10, 20], k=1),
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        k=1
                    ),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': {
                    'sizes': [10],
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        k=1
                    ),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt'],
                    'forberg': ['ecg_0']
                },
            },
            pre_processor=sklearn_process,
            pre_processor_kwargs={
                'flat_features': {'processor': StandardScaler},
                'forberg_ecg_0': {'processor': StandardScaler},
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
                }
            },
            epochs=100,
            batch_size=64,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1 / 3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=44,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )
    M_F2_FF_NN_RS = HyperExperiment(
        template=Experiment(
            description="Random search over simple feed-forward neural "
                        "networks, using forberg features from 2 ECGs, and "
                        "flat-features. ",
            model=ffnn,
            model_kwargs={
                'ecg_ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [25, 50, 100, 200],
                            k=num_layers,
                            ascending=False,
                        ),
                        'dropouts': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'batch_norms': hp.Choices(
                            [True, False],
                            k=num_layers
                        ),
                        'activity_regularizers': hp.Choices(
                            [0.01, 0.001, 0.0001, 0.00001, 0.0],
                            k=num_layers
                        ),
                        'kernel_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        ),
                        'bias_regularizers': hp.Choices(
                            [0.1, 0.01, 0.001, 0.0001, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2]
                ]),
                'ecg_combiner': hp.Choice(['concatenate', 'difference']),
                'ecg_comb_ffnn_kwargs': {
                    'sizes': hp.Choices([6, 10, 20], k=1),
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        k=1
                    ),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
                'flat_ffnn_kwargs': None,
                'final_ffnn_kwargs': {
                    'sizes': [10],
                    'dropouts': hp.Choices(
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        k=1
                    ),
                    'batch_norms': hp.Choices([True, False], k=1),
                    'activity_regularizers': hp.Choices(
                        [0.01, 0.001, 0.0001, 0.00001, 0.0],
                        k=1
                    ),
                    'kernel_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    ),
                    'bias_regularizers': hp.Choices(
                        [0.1, 0.01, 0.001, 0.0001, 0.0],
                        k=1
                    )
                },
            },
            extractor=EscTrop,
            extractor_kwargs={
                'features': {
                    'flat_features': ['log_tnt_1', 'age', 'male', 'log_dt'],
                    'forberg': ['ecg_0', 'ecg_1']
                },
            },
            pre_processor=sklearn_process,
            pre_processor_kwargs={
                'flat_features': {'processor': StandardScaler},
                'forberg_ecg_0': {'processor': StandardScaler},
                'forberg_ecg_1': {'processor': StandardScaler},
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
                    ])
                }
            },
            epochs=100,
            batch_size=64,
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 1 / 3
            },
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            random_state=hp.Int(0, 1000000000),
        ),
        random_seed=45,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 400
        },
    )
