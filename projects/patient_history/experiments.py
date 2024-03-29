from enum import Enum

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Binarizer, StandardScaler
from xgboost import XGBClassifier
from keras.optimizers import Adam
from keras.losses import BinaryFocalCrossentropy, BinaryCrossentropy

from mim.experiments.experiments import Experiment
from mim.experiments.hyper_experiments import HyperExperiment
from mim.experiments.search_strategies import RandomSearch
from mim.experiments import hyper_parameter as hp
from mim.experiments.extractor import sklearn_process
from projects.patient_history.extractor import Flat
from projects.patient_history.models import mlp1, mlp2


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
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler}
        },
        scoring=roc_auc_score,
        metrics=['accuracy', 'auc'],
    )
    LR_LISA = LR_BASIC._replace(
        description='Predicting ACS or death using data from LISA',
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
        extractor_kwargs={
            'features': {'lisa': {}}
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'lisa': {'processor': StandardScaler},
        },
    )
    LR_LISA_BASIC = LR_BASIC._replace(
        description='Predicting ACS or death using data from LISA',
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.01,
        },
        extractor_kwargs={
            'features': {
                'lisa': {},
                'basic': ['age', 'sex']
            }
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'lisa': {'processor': StandardScaler},
            'basic': {'processor': StandardScaler}
        },
    )
    LR_LISA_FAM_BASIC = LR_LISA_BASIC._replace(
        description='Predicting ACS or death using data from LISA',
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.01
        },
        extractor_kwargs={
            'features': {
                'lisa': {'family': True},
                'basic': ['age', 'sex']
            }
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler}
        },
    )
    LR_LISA_EDU_BASIC = LR_LISA_BASIC._replace(
        description='Predicting ACS or death using data from LISA',
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.01
        },
        extractor_kwargs={
            'features': {
                'lisa': {'education': True},
                'basic': ['age', 'sex']
            }
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler}
        },
    )
    LR_LISA_OCC_BASIC = LR_LISA_BASIC._replace(
        description='Predicting ACS or death using data from LISA',
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.01
        },
        extractor_kwargs={
            'features': {
                'lisa': {'occupation': True},
                'basic': ['age', 'sex']
            }
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler}
        },
    )
    LR_LISA_INC_BASIC = LR_LISA_BASIC._replace(
        description='Predicting ACS or death using data from LISA',
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.01
        },
        extractor_kwargs={
            'features': {
                'lisa': {'income': True},
                'basic': ['age', 'sex']
            }
        },
    )

    LR_SI10_P1 = Experiment(
        description='',
        model=LogisticRegression,
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 1.0
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
                'lisa': {},
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
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
    LR_SI1000_OI1000_P1 = LR_SIC_BASIC_P1._replace(
        description='',
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV', 'OV'],
                    'num_icd': 1000,
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_SK1000_OK1000_P1 = LR_SIC_BASIC_P1._replace(
        description='',
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV', 'OV'],
                    'num_icd': 0,
                    'num_kva': 1000,
                    'num_atc': 0,
                }
            }
        },
    )

    LR_A1000_I1000_K100 = Experiment(
        description='This input set represents the best performing subset '
                    'of each individual source, with respect to the LR '
                    'model.',
        model=LogisticRegression,
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
        extractor=Flat,
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV', 'OV'],
                    'num_icd': 1000,
                    'num_kva': 100,
                    'num_atc': 1000,
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
    LR_I1000 = LR_A1000_I1000_K100._replace(
        description="Combines ICD codes from SV and OV",
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV', 'OV'],
                    'num_icd': 1000,
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_K100 = LR_A1000_I1000_K100._replace(
        description="Combines KVÅ codes from SV and OV",
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV', 'OV'],
                    'num_icd': 0,
                    'num_kva': 100,
                    'num_atc': 0,
                }
            }
        },
    )
    LR_AIK_BASIC = LR_A1000_I1000_K100._replace(
        description="AIK refers to the best combination of ATC, ICD and KVÅ "
                    "codes found so far.",
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV', 'OV'],
                    'num_icd': 1000,
                    'num_kva': 100,
                    'num_atc': 1000,
                }
            }
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler}
        },
    )
    LR_AIK_BASIC_50 = LR_A1000_I1000_K100._replace(
        description="AIK refers to the best combination of ATC, ICD and KVÅ "
                    "codes found so far. Only patients over 50 years.",
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
        extractor_kwargs={
            'index': {'age_threshold': 50},
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV', 'OV'],
                    'num_icd': 1000,
                    'num_kva': 100,
                    'num_atc': 1000,
                }
            }
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler}
        },
    )
    LR_AIK_LISA = LR_A1000_I1000_K100._replace(
        description="",
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
        extractor_kwargs={
            'features': {
                'lisa': {},
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV', 'OV'],
                    'num_icd': 1000,
                    'num_kva': 100,
                    'num_atc': 1000,
                }
            }
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'lisa': {'processor': StandardScaler}
        },
    )
    LR_AIK_LISA_BASIC = LR_A1000_I1000_K100._replace(
        description="",
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
        extractor_kwargs={
            'features': {
                'lisa': {},
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV', 'OV'],
                    'num_icd': 1000,
                    'num_kva': 100,
                    'num_atc': 1000,
                }
            }
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler},
            'lisa': {'processor': StandardScaler}
        },
    )
    LR_A1000_BASIC = LR_A1000_I1000_K100._replace(
        description="",
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'num_icd': 0,
                    'num_kva': 0,
                    'num_atc': 1000,
                }
            }
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler},
        },
    )
    LR_I1000_BASIC = LR_A1000_I1000_K100._replace(
        description="",
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV', 'OV'],
                    'num_icd': 1000,
                    'num_kva': 0,
                    'num_atc': 0,
                }
            }
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler},
        },
    )
    LR_K100_BASIC = LR_A1000_I1000_K100._replace(
        description="",
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV', 'OV'],
                    'num_icd': 0,
                    'num_kva': 100,
                    'num_atc': 0,
                }
            }
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler},
        },
    )

    MLP_BASIC = Experiment(
        description='Basic experiment with only age and sex as features.',
        model=mlp2,
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [10, 5],
                'dropout': [0.0, 0.0],
                'regularizer': {
                    'kernel': 1e-2,
                    'activity': 0.0,
                    'bias': 0.0,
                }
            },
        },
        extractor=Flat,
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
            }
        },
        building_model_requires_development_data=True,
        batch_size=256,
        epochs=100,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-2}
        },
        loss=BinaryCrossentropy,
        loss_kwargs={},
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler},
        },
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43,
        },
        scoring=roc_auc_score,
        metrics=['accuracy', 'auc'],
        rule_out_logger=True,
    )
    MLP_BASIC_W1 = MLP_BASIC._replace(
        description='Cross-entropy, but with class weights.',
        batch_size=256,
        epochs=100,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-2}
        },
        loss=BinaryCrossentropy,
        loss_kwargs={},
        class_weight={1: 10.0, 0: 1.0}
    )
    MLP_BASIC_W2 = MLP_BASIC._replace(
        description='Cross-entropy, but with more skewed class weights.',
        batch_size=256,
        epochs=100,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-2}
        },
        loss=BinaryCrossentropy,
        loss_kwargs={},
        class_weight={1: 100.0, 0: 1.0}
    )
    MLP_BASIC_W3 = MLP_BASIC._replace(
        description='Cross-entropy, but with class weights inverted.',
        batch_size=256,
        epochs=100,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-2}
        },
        loss=BinaryCrossentropy,
        loss_kwargs={},
        class_weight={1: 1.0, 0: 100.0}
    )
    MLP_BASIC_BFE1 = MLP_BASIC._replace(
        description='Binary focal cross-entropy, default parameters. Seems '
                    'like I need lower learning-rate for this to work at all.',
        batch_size=256,
        epochs=100,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-3}
        },
        loss=BinaryFocalCrossentropy,
        loss_kwargs={
            'apply_class_balancing': False,
            'alpha': 0.25,
            'gamma': 2.0
        },
    )
    MLP_BASIC_BFE2 = MLP_BASIC._replace(
        description='Binary focal cross-entropy, higher alpha.',
        batch_size=256,
        epochs=100,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-3}
        },
        loss=BinaryFocalCrossentropy,
        loss_kwargs={
            'apply_class_balancing': False,
            'alpha': 0.50,
            'gamma': 2.0
        },
    )
    MLP_BASIC_BFE3 = MLP_BASIC._replace(
        description='Binary focal cross-entropy, higher gamma.',
        batch_size=256,
        epochs=100,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-3}
        },
        loss=BinaryFocalCrossentropy,
        loss_kwargs={
            'apply_class_balancing': False,
            'alpha': 0.25,
            'gamma': 4.0
        },
    )
    MLP_BASIC_BFE4 = MLP_BASIC._replace(
        description='Binary focal cross-entropy, class-balancing.',
        batch_size=256,
        epochs=100,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-3}
        },
        loss=BinaryFocalCrossentropy,
        loss_kwargs={
            'apply_class_balancing': True,
            'alpha': 0.25,
            'gamma': 2.0
        },
    )
    # MLP_BASIC_BFC_V1 = MLP_BASIC._replace(
    #     description='Binary focal cross-entropy, version 1.',
    #     batch_size=256,
    #     epochs=100,
    #     optimizer={
    #         'name': Adam,
    #         'kwargs': {'learning_rate': 1e-2}
    #     },
    #     loss=BinaryFocalCrossentropy,
    #     loss_kwargs={},
    # )

    MLP1_AC_SIC_OIC_BASIC = Experiment(
        description='ICD codes from SV grouped into chapters.',
        model=mlp1,
        model_kwargs={
            'history_mlp_kwargs': {
                'sizes': [500, 10],
                'dropout': [0.5, 0.2],
                'regularizer': 1e-3,
            },
            'final_mlp_kwargs': {
                'sizes': [10],
                'dropout': [0.0]
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

    MLP1_LISA = Experiment(
        description='',
        model=mlp1,
        model_kwargs={
            'lisa_mlp_kwargs': {
                'sizes': [100, 10],
                'dropout': [0.5, 0.2],
                'regularizer': 1e-3,
            },
            'final_mlp_kwargs': {
                'sizes': [10],
                'dropout': [0.0]
            }
        },
        extractor=Flat,
        extractor_kwargs={
            'features': {
                'lisa': {},
            }
        },
        building_model_requires_development_data=True,
        batch_size=256,
        epochs=100,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-3}
        },
        pre_processor_kwargs={
            'lisa': {'processor': StandardScaler}
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

    MLP1_LISA_BASIC = MLP1_LISA._replace(
        description='',
        model_kwargs={
            'lisa_mlp_kwargs': {
                'sizes': [100, 10],
                'dropout': [0.5, 0.2],
                'regularizer': 1e-2,
            },
            'final_mlp_kwargs': {
                'sizes': [10],
                'dropout': [0.0]
            }
        },
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'lisa': {},
            }
        },
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler},
            'lisa': {'processor': StandardScaler}
        },
    )

    MLP1_LISA_BASIC_ENSEMBLE = MLP1_LISA._replace(
        description='Repeats the same model 500 times, to investigate the '
                    'effect on the rule-out metric.',
        model_kwargs={
            'lisa_mlp_kwargs': {
                'sizes': [100, 10],
                'dropout': [0.5, 0.2],
                'regularizer': 1e-2,
            },
            'final_mlp_kwargs': {
                'sizes': [10],
                'dropout': [0.0]
            }
        },
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'lisa': {},
            }
        },
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler},
            'lisa': {'processor': StandardScaler}
        },
        epochs=100,
        save_model=False,
        ensemble=500,
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43,
        },
    )

    MLP1_AIK_LISA_BASIC = MLP1_LISA._replace(
        description='',
        model_kwargs={
            'history_mlp_kwargs': {
                'sizes': [100, 10],
                'dropout': [0.5, 0.2],
                'regularizer': 1e-2,
            },
            'lisa_mlp_kwargs': {
                'sizes': [100, 10],
                'dropout': [0.5, 0.2],
                'regularizer': 1e-2,
            },
            'final_mlp_kwargs': {
                'sizes': [10],
                'dropout': [0.0]
            }
        },
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'lisa': {},
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV', 'SV'],
                    'num_icd': 1000,
                    'num_atc': 1000,
                    'num_kva': 100,
                }
            }
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler},
            'lisa': {'processor': StandardScaler}
        },
    )
    MLP1_AIK_BASIC = MLP1_LISA._replace(
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV', 'SV'],
                    'num_icd': 1000,
                    'num_atc': 1000,
                    'num_kva': 100,
                }
            }
        },
        model_kwargs={
            'history_mlp_kwargs': {
                'sizes': [100, 10],
                'dropout': [0.5, 0.2],
                'regularizer': 1e-2,
            },
            'final_mlp_kwargs': {
                'sizes': [10],
                'dropout': [0.0]
            }
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler},
        },
    )

    # Best models after random search, so far (177, 65, 21 and 31 iterations):
    MLP2_BLAIK = Experiment(
        description='All the data. Model is xp_84 from the random search.',
        model=mlp2,
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [500, 100, 50],
                'dropout': [0.2, 0.5, 0.5],
                'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
        extractor=Flat,
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'lisa': {},
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV', 'SV'],
                    'num_icd': 1000,
                    'num_atc': 1000,
                    'num_kva': 100,
                }
            }
        },
        building_model_requires_development_data=True,
        batch_size=256,
        epochs=200,
        ensemble=10,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-2}
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler},
            'lisa': {'processor': StandardScaler}
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
    MLP2_BAIK = MLP2_BLAIK._replace(
        description='Age+sex+ATC+ICD+KVÅ. Model is xp_59 from the random '
                    'search',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [100, 100, 10],
                'dropout': [0.1, 0.2, 0.4],
                'regularizer': [1e-2, 1e-3, 0.0],
            },
        },
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV', 'SV'],
                    'num_icd': 1000,
                    'num_atc': 1000,
                    'num_kva': 100,
                }
            }
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-2}
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler},
        },
    )
    MLP2_LAIK = MLP2_BLAIK._replace(
        description='LISA+ATC+ICD+KVÅ. Model is xp_19 from the random '
                    'search.',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [100],
                'dropout': [0.5],
                'regularizer': [1e-3],
            },
        },
        extractor_kwargs={
            'features': {
                'lisa': {},
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV', 'SV'],
                    'num_icd': 1000,
                    'num_atc': 1000,
                    'num_kva': 100,
                }
            }
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 3e-3}
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'lisa': {'processor': StandardScaler}
        },
    )
    MLP2_BL = MLP2_BLAIK._replace(
        description='Age+sex+LISA. Model is xp_4 from the random search.',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [50, 10],
                'dropout': [0.3, 0.5],
                'regularizer': [1e-3, 1e-4],
            },
        },
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'lisa': {},
            }
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-4}
        },
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler},
            'lisa': {'processor': StandardScaler}
        },
    )

    # Best models after full random search (200 iterations each, except BAIK
    # which has only 157 iterations):
    MLP3_BLAIK = Experiment(
        description='All the data. Model is xp_142 from the random search.',
        model=mlp2,
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [500, 50, 50],
                'dropout': [0.5, 0.4, 0.1],
                'regularizer': [1e-2, 1e-4, 1e-3],
            },
        },
        extractor=Flat,
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'lisa': {},
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV', 'SV'],
                    'num_icd': 1000,
                    'num_atc': 1000,
                    'num_kva': 100,
                }
            }
        },
        building_model_requires_development_data=True,
        batch_size=256,
        epochs=200,
        ensemble=10,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-3}
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler},
            'lisa': {'processor': StandardScaler}
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
    MLP3_BAIK = MLP3_BLAIK._replace(
        description='Age+sex+ATC+ICD+KVÅ. Model is xp_84 from the random '
                    'search',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [500, 100, 50],
                'dropout': [0.2, 0.2, 0.5],
                'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV', 'SV'],
                    'num_icd': 1000,
                    'num_atc': 1000,
                    'num_kva': 100,
                }
            }
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-2}
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler},
        },
    )
    MLP3_LAIK = MLP3_BLAIK._replace(
        description='LISA+ATC+ICD+KVÅ. Model is xp_84 from the random '
                    'search.',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [500, 100, 50],
                'dropout': [0.2, 0.2, 0.5],
                'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
        extractor_kwargs={
            'features': {
                'lisa': {},
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV', 'SV'],
                    'num_icd': 1000,
                    'num_atc': 1000,
                    'num_kva': 100,
                }
            }
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-2}
        },
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'lisa': {'processor': StandardScaler}
        },
    )
    MLP3_BL = MLP3_BLAIK._replace(
        description='Age+sex+LISA. Model is xp_152 from the random search.',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [500],
                'dropout': [0.3],
                'regularizer': [1e-4],
            },
        },
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'lisa': {},
            }
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 3e-4}
        },
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler},
            'lisa': {'processor': StandardScaler}
        },
    )

    # Other models
    RF_BASIC = Experiment(
        description='Predicting ACS or death using only age and sex',
        model=RandomForestClassifier,
        model_kwargs={
            'class_weight': 'balanced',
            'n_estimators': 1000,
            'n_jobs': -1
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
                'lisa': {},
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
                'lisa': {},
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


class HyperSearch(HyperExperiment, Enum):
    MLP2_LAIK_BASIC = HyperExperiment(
        template=Experiment(
            description='LISA + ATC + ICD + KVÅ + age + sex to predict ACS',
            model=mlp2,
            model_kwargs={
                'mlp_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [500, 100, 50, 10],
                            k=num_layers,
                            ascending=False
                        ),
                        'dropout': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'regularizer': hp.Choices(
                            [1e-2, 1e-3, 1e-4, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2, 3]
                ]),
            },
            extractor=Flat,
            extractor_kwargs={
                'features': {
                    'basic': ['age', 'sex'],
                    'lisa': {},
                    'history': {
                        'intervals': {'periods': 1},
                        'sources': ['OV', 'SV'],
                        'num_icd': 1000,
                        'num_atc': 1000,
                        'num_kva': 100,
                    }
                }
            },
            pre_processor=sklearn_process,
            pre_processor_kwargs={
                'history': {'processor': Binarizer},
                'lisa': {'processor': StandardScaler},
                'basic': {'processor': StandardScaler},
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        1e-2, 3e-3, 1e-3, 3e-4, 1e-4
                    ])
                }
            },
            batch_size=256,
            epochs=200,
            ensemble=10,
            cv=GroupShuffleSplit,
            cv_kwargs={
                'n_splits': 1,
                'train_size': 2 / 3,
                'random_state': 43,
            },
            scoring=roc_auc_score,
            metrics=['accuracy', 'auc'],
            building_model_requires_development_data=True,
            save_model=False,
            random_state=hp.Int(0, 1000000000)
        ),
        random_seed=42,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 200
        }
    )

    MLP2_AIK_BASIC = HyperExperiment(
        template=Experiment(
            description='ATC + ICD + KVÅ + age + sex to predict ACS',
            model=mlp2,
            model_kwargs={
                'mlp_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [500, 100, 50, 10],
                            k=num_layers,
                            ascending=False
                        ),
                        'dropout': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'regularizer': hp.Choices(
                            [1e-2, 1e-3, 1e-4, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2, 3]
                ]),
            },
            extractor=Flat,
            extractor_kwargs={
                'features': {
                    'basic': ['age', 'sex'],
                    'history': {
                        'intervals': {'periods': 1},
                        'sources': ['OV', 'SV'],
                        'num_icd': 1000,
                        'num_atc': 1000,
                        'num_kva': 100,
                    }
                }
            },
            pre_processor=sklearn_process,
            pre_processor_kwargs={
                'history': {'processor': Binarizer},
                'basic': {'processor': StandardScaler},
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        1e-2, 3e-3, 1e-3, 3e-4, 1e-4
                    ])
                }
            },
            batch_size=256,
            epochs=200,
            ensemble=10,
            cv=GroupShuffleSplit,
            cv_kwargs={
                'n_splits': 1,
                'train_size': 2 / 3,
                'random_state': 43,
            },
            scoring=roc_auc_score,
            metrics=['accuracy', 'auc'],
            building_model_requires_development_data=True,
            save_model=False,
            random_state=hp.Int(0, 1000000000)
        ),
        random_seed=42,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 200
        }
    )

    MLP2_LISA_BASIC = HyperExperiment(
        template=Experiment(
            description='LISA + age + sex to predict ACS',
            model=mlp2,
            model_kwargs={
                'mlp_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [500, 100, 50, 10],
                            k=num_layers,
                            ascending=False
                        ),
                        'dropout': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'regularizer': hp.Choices(
                            [1e-2, 1e-3, 1e-4, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2, 3]
                ]),
            },
            extractor=Flat,
            extractor_kwargs={
                'features': {
                    'basic': ['age', 'sex'],
                    'lisa': {},
                }
            },
            pre_processor=sklearn_process,
            pre_processor_kwargs={
                'lisa': {'processor': StandardScaler},
                'basic': {'processor': StandardScaler},
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        1e-2, 3e-3, 1e-3, 3e-4, 1e-4
                    ])
                }
            },
            batch_size=256,
            epochs=200,
            ensemble=10,
            cv=GroupShuffleSplit,
            cv_kwargs={
                'n_splits': 1,
                'train_size': 2 / 3,
                'random_state': 43,
            },
            scoring=roc_auc_score,
            metrics=['accuracy', 'auc'],
            building_model_requires_development_data=True,
            save_model=False,
            random_state=hp.Int(0, 1000000000)
        ),
        random_seed=42,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 200
        }
    )

    MLP2_LAIK = HyperExperiment(
        template=Experiment(
            description='LISA + ATC + ICD + KVÅ to predict ACS',
            model=mlp2,
            model_kwargs={
                'mlp_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [500, 100, 50, 10],
                            k=num_layers,
                            ascending=False
                        ),
                        'dropout': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'regularizer': hp.Choices(
                            [1e-2, 1e-3, 1e-4, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2, 3]
                ]),
            },
            extractor=Flat,
            extractor_kwargs={
                'features': {
                    'lisa': {},
                    'history': {
                        'intervals': {'periods': 1},
                        'sources': ['OV', 'SV'],
                        'num_icd': 1000,
                        'num_atc': 1000,
                        'num_kva': 100,
                    }
                }
            },
            pre_processor=sklearn_process,
            pre_processor_kwargs={
                'history': {'processor': Binarizer},
                'lisa': {'processor': StandardScaler},
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        1e-2, 3e-3, 1e-3, 3e-4, 1e-4
                    ])
                }
            },
            batch_size=256,
            epochs=200,
            ensemble=10,
            cv=GroupShuffleSplit,
            cv_kwargs={
                'n_splits': 1,
                'train_size': 2 / 3,
                'random_state': 43,
            },
            scoring=roc_auc_score,
            metrics=['accuracy', 'auc'],
            building_model_requires_development_data=True,
            save_model=False,
            random_state=hp.Int(0, 1000000000)
        ),
        random_seed=42,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 200
        }
    )

    MLP2_BLAIK_OVER50 = HyperExperiment(
        template=Experiment(
            description='LISA + age + sex to predict ACS for patients '
                        '50 years or older.',
            model=mlp2,
            model_kwargs={
                'mlp_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [500, 100, 50, 10],
                            k=num_layers,
                            ascending=False
                        ),
                        'dropout': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'regularizer': hp.Choices(
                            [1e-2, 1e-3, 1e-4, 0.0],
                            k=num_layers
                        )
                    } for num_layers in [1, 2, 3]
                ]),
            },
            extractor=Flat,
            extractor_kwargs={
                'features': {
                    'basic': ['age', 'sex'],
                    'lisa': {},
                }
            },
            pre_processor=sklearn_process,
            pre_processor_kwargs={
                'lisa': {'processor': StandardScaler},
                'basic': {'processor': StandardScaler},
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([
                        1e-2, 3e-3, 1e-3, 3e-4, 1e-4
                    ])
                }
            },
            batch_size=256,
            epochs=200,
            ensemble=10,
            cv=GroupShuffleSplit,
            cv_kwargs={
                'n_splits': 1,
                'train_size': 2 / 3,
                'random_state': 43,
            },
            scoring=roc_auc_score,
            metrics=['accuracy', 'auc'],
            building_model_requires_development_data=True,
            save_model=False,
            random_state=hp.Int(0, 1000000000)
        ),
        random_seed=42,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 200
        }
    )
