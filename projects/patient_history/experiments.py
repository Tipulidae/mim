from enum import Enum

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer, StandardScaler

from mim.experiments.experiments import Experiment
from mim.experiments.extractor import sklearn_process
from projects.patient_history.extractor import Flat


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
