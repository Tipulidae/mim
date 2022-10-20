from enum import Enum

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Binarizer, StandardScaler, PowerTransformer
from xgboost import XGBClassifier
from tensorflow.keras.optimizers import Adam

from mim.experiments.experiments import Experiment
from mim.experiments.extractor import sklearn_process
from projects.patient_history.extractor import Flat
from projects.patient_history.models import simple_mlp


class PatientHistory(Experiment, Enum):
    LR_BASIC = Experiment(
        description='Predicting ACS or death using only age and sex',
        model=LogisticRegression,
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300
        },
        extractor=Flat,
        extractor_kwargs={
            'features': {'basic': ['age', 'sex']}
        },
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43,
        },
        scoring=roc_auc_score,
        metrics=['accuracy', 'auc'],
    )
    LR_LISA = LR_BASIC._replace(
        description='Predicting ACS or death using data from LISA',
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.0001
        },
        extractor_kwargs={
            'features': {'lisa': True}
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'lisa': {'processor': PowerTransformer},
        },
    )
    LR_LISA_BASIC = LR_BASIC._replace(
        description='Predicting ACS or death using data from LISA',
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.01
        },
        extractor_kwargs={
            'features': {
                'lisa': True,
                'basic': ['age', 'sex']
            }
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'lisa': {'processor': PowerTransformer},
            'basic': {'processor': StandardScaler}
        },
    )

    LR_SI10_P1 = Experiment(
        description='',
        model=LogisticRegression,
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300
        },
        extractor=Flat,
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV'],
                    'num_icd': 10,
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
        },
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43,
        },
        scoring=roc_auc_score,
        metrics=['accuracy', 'auc'],
    )
    LR_SI10_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'sources': ['SV'],
                    'num_icd': 10,
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_SI100_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV'],
                    'num_icd': 100,
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_SI100_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'sources': ['SV'],
                    'num_icd': 100,
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_SI1000_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV'],
                    'num_icd': 1000,
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_SI1000_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'sources': ['SV'],
                    'num_icd': 1000,
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
    )

    LR_SK10_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV'],
                    'num_icd': 0,
                    'num_kva': 10,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_SK10_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'sources': ['SV'],
                    'num_icd': 0,
                    'num_kva': 10,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_SK100_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV'],
                    'num_icd': 0,
                    'num_kva': 100,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_SK100_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'sources': ['SV'],
                    'num_icd': 0,
                    'num_kva': 100,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_SK1000_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV'],
                    'num_icd': 0,
                    'num_kva': 1000,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_SK1000_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'sources': ['SV'],
                    'num_icd': 0,
                    'num_kva': 1000,
                    'num_atc': 0,
                }
            }
        },
    )

    LR_OI10_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV'],
                    'num_icd': 10,
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_OI10_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'sources': ['OV'],
                    'num_icd': 10,
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_OI100_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV'],
                    'num_icd': 100,
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_OI100_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'sources': ['OV'],
                    'num_icd': 100,
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_OI1000_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV'],
                    'num_icd': 1000,
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_OI1000_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'sources': ['OV'],
                    'num_icd': 1000,
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
    )

    LR_OK10_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV'],
                    'num_icd': 0,
                    'num_kva': 10,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_OK10_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'sources': ['OV'],
                    'num_icd': 0,
                    'num_kva': 10,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_OK100_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV'],
                    'num_icd': 0,
                    'num_kva': 100,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_OK100_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'sources': ['OV'],
                    'num_icd': 0,
                    'num_kva': 100,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_OK1000_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV'],
                    'num_icd': 0,
                    'num_kva': 1000,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_OK1000_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'sources': ['OV'],
                    'num_icd': 0,
                    'num_kva': 1000,
                    'num_atc': 0,
                }
            }
        },
    )

    LR_A10_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'num_icd': 0,
                    'num_kva': 0,
                    'num_atc': 10,
                }
            }
        },
    )

    LR_A10_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'num_icd': 0,
                    'num_kva': 0,
                    'num_atc': 10,
                }
            }
        },
    )
    LR_A100_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'num_icd': 0,
                    'num_kva': 0,
                    'num_atc': 100,
                }
            }
        },
    )
    LR_A100_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'num_icd': 0,
                    'num_kva': 0,
                    'num_atc': 100,
                }
            }
        },
    )
    LR_A1000_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'num_icd': 0,
                    'num_kva': 0,
                    'num_atc': 1000,
                }
            }
        },
    )
    LR_A1000_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'num_icd': 0,
                    'num_kva': 0,
                    'num_atc': 1000,
                }
            }
        },
    )

    # ROUND TWO
    LR_SIC_P1 = Experiment(
        description='ICD codes from SV grouped into chapters.',
        model=LogisticRegression,
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300
        },
        extractor=Flat,
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV'],
                    'num_icd': -1,
                    'icd_level': 'chapter',
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
        },
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43,
        },
        scoring=roc_auc_score,
        metrics=['accuracy', 'auc'],
    )
    LR_SIC_BASIC_P1 = LR_SIC_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV'],
                    'num_icd': -1,
                    'icd_level': 'chapter',
                    'num_kva': 0,
                    'num_atc': 0,
                }

            }
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler}
        },
    )
    LR_OIC_P1 = LR_SIC_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV'],
                    'num_icd': -1,
                    'icd_level': 'chapter',
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_OIC_BASIC_P1 = LR_SIC_BASIC_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV'],
                    'num_icd': -1,
                    'icd_level': 'chapter',
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_AC_P1 = LR_SIC_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'num_icd': 0,
                    'atc_level': 'therapeutic',
                    'num_kva': 0,
                    'num_atc': -1,
                }
            }
        },
    )
    LR_AC_BASIC_P1 = LR_SIC_BASIC_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'num_icd': 0,
                    'atc_level': 'therapeutic',
                    'num_kva': 0,
                    'num_atc': -1,
                }
            }
        },
    )
    LR_AC_SIC_OIC_P1 = LR_SIC_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV', 'SV'],
                    'num_icd': -1,
                    'icd_level': 'chapter',
                    'num_atc': -1,
                    'atc_level': 'therapeutic',
                    'num_kva': 0,
                }
            }
        },
    )
    LR_AC_SIC_OIC_BASIC_P1 = LR_SIC_BASIC_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV', 'SV'],
                    'num_icd': -1,
                    'icd_level': 'chapter',
                    'num_atc': -1,
                    'atc_level': 'therapeutic',
                    'num_kva': 0,
                }
            }
        },
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },

    )
    LR_AC_SIC_OIC_LISA_BASIC_P1 = LR_SIC_BASIC_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'lisa': True,
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV', 'SV'],
                    'num_icd': -1,
                    'icd_level': 'chapter',
                    'num_atc': -1,
                    'atc_level': 'therapeutic',
                    'num_kva': 0,
                }
            }
        },
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler},
            'lisa': {'processor': StandardScaler}
        },
    )

    LR_A100_SI100_OI100_P1 = LR_SIC_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV', 'OV'],
                    'num_icd': 100,
                    'num_kva': 0,
                    'num_atc': 100,
                }
            }
        },
    )
    LR_A100_SI100_OI100_BASIC_P1 = LR_SIC_BASIC_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV', 'OV'],
                    'num_icd': 100,
                    'num_kva': 0,
                    'num_atc': 100,
                }
            }
        },
    )

    MLP1_AC_SIC_OIC_BASIC = Experiment(
        description='ICD codes from SV grouped into chapters.',
        model=simple_mlp,
        model_kwargs={
            'history_mlp_kwargs': {
                'sizes': [500, 10],
                'dropouts': [0.5, 0.2],
                'default_regularizer': 1e-3,
            },
            'final_mlp_kwargs': {
                'sizes': [10],
                'dropouts': [0.0]
            }
        },
        extractor=Flat,
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV', 'SV'],
                    'num_icd': -1,
                    'icd_level': 'chapter',
                    'num_atc': -1,
                    'atc_level': 'therapeutic',
                    'num_kva': 0,
                }
            }
        },
        building_model_requires_development_data=True,
        batch_size=256,
        epochs=100,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-3}
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler}
        },
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43,
        },
        scoring=roc_auc_score,
        metrics=['accuracy', 'auc'],
    )
    MLP1_A1000_SI1000_OI1000_BASIC = MLP1_AC_SIC_OIC_BASIC._replace(
        model_kwargs={
            'history_mlp_kwargs': {
                'sizes': [100, 10],
                'dropouts': [0.5, 0.2],
                'default_regularizer': 1e-2,
            },
            'final_mlp_kwargs': {
                'sizes': [10],
                'dropouts': [0.0]
            }
        },
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV', 'SV'],
                    'num_icd': 1000,
                    'num_atc': 1000,
                    'num_kva': 0,
                }
            }
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler},
        },
    )

    MLP1_LISA = MLP1_AC_SIC_OIC_BASIC._replace(
        description='',
        model_kwargs={
            'lisa_mlp_kwargs': {
                'sizes': [100, 10],
                'dropouts': [0.5, 0.2],
                'default_regularizer': 1e-2,
            },
            'final_mlp_kwargs': {
                'sizes': [10],
                'dropouts': [0.0]
            }
        },
        extractor_kwargs={
            'features': {
                'lisa': True,
            }
        },
        pre_processor_kwargs={
            'lisa': {'processor': StandardScaler}
        },
    )
    MLP1_LISA2 = MLP1_LISA._replace(
        random_state=1239823
    )

    MLP1_LISA_BASIC = MLP1_AC_SIC_OIC_BASIC._replace(
        description='',
        model_kwargs={
            'lisa_mlp_kwargs': {
                'sizes': [100, 10],
                'dropouts': [0.5, 0.2],
                'default_regularizer': 1e-2,
            },
            'final_mlp_kwargs': {
                'sizes': [10],
                'dropouts': [0.0]
            }
        },
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'lisa': True,
            }
        },
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler},
            'lisa': {'processor': StandardScaler}
        },
        save_prediction_history=True

        # metrics=['auc', 'rule_out']
        # scoring=rule_out
    )
    MLP1_LISA_BASIC2 = MLP1_LISA_BASIC._replace(
        description='',
        random_state=12393,
        save_prediction_history=False
        # save_prediction_history=True

        # metrics=['auc', 'rule_out']
        # scoring=rule_out
    )

    MLP1_LISA_BASIC_W = MLP1_AC_SIC_OIC_BASIC._replace(
        description='',
        model_kwargs={
            'lisa_mlp_kwargs': {
                'sizes': [100, 10],
                'dropouts': [0.5, 0.2],
                'default_regularizer': 1e-2,
            },
            'final_mlp_kwargs': {
                'sizes': [10],
                'dropouts': [0.0]
            }
        },
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'lisa': True,
            }
        },
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler},
            'lisa': {'processor': StandardScaler}
        },
        loss_weights=[1e-3, 1.0],
        # save_prediction_history=True
    )

    MLP1_A1000_SI1000_OI1000_LISA_BASIC = MLP1_AC_SIC_OIC_BASIC._replace(
        description='',
        model_kwargs={
            'history_mlp_kwargs': {
                'sizes': [100, 10],
                'dropouts': [0.5, 0.2],
                'default_regularizer': 1e-2,
            },
            'lisa_mlp_kwargs': {
                'sizes': [100, 10],
                'dropouts': [0.5, 0.2],
                'default_regularizer': 1e-2,
            },
            'final_mlp_kwargs': {
                'sizes': [10],
                'dropouts': [0.0]
            }
        },
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'lisa': True,
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV', 'SV'],
                    'num_icd': 1000,
                    'num_atc': 1000,
                    'num_kva': 0,
                }
            }
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler},
            'lisa': {'processor': StandardScaler}
        },
    )

    RF_BASIC = Experiment(
        description='Predicting ACS or death using only age and sex',
        model=RandomForestClassifier,
        model_kwargs={
            'class_weight': 'balanced',
            'n_estimators': 1000,
            'n_jobs': -1
            # 'max_iter': 300
        },
        extractor=Flat,
        extractor_kwargs={
            'features': {'basic': ['age', 'sex']}
        },
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43,
        },
        scoring=roc_auc_score,
        metrics=['accuracy', 'auc'],
    )

    RF_LISA_BASIC = RF_BASIC._replace(
        description='',
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'lisa': True,
            }
        },
    )

    XGB_LISA_BASIC = Experiment(
        description='Predicting ACS or death using only age and sex',
        model=XGBClassifier,
        model_kwargs={
        },
        extractor=Flat,
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'lisa': True,
            }
        },
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43,
        },
        scoring=roc_auc_score,
        metrics=['accuracy', 'auc'],
    )
