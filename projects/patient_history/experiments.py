from enum import Enum

from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.decomposition import PCA

from mim.experiments.experiments import Experiment
from mim.experiments.extractor import sklearn_process
from projects.patient_history.extractor import Flat, Ragged
from projects.patient_history.models import simple_lstm, simple_ffnn


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
        # pre_processor=sklearn_process,
        # pre_processor_kwargs={
        #     'history': {'processor': Binarizer},
        # },
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43,
        },
        scoring=roc_auc_score,
        metrics=['accuracy', 'auc'],
    )
    LR_AC_SIC_OIC_BASIC_P1 = LR_BASIC._replace(
        description='',
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'diagnoses': -1,
                    'interventions': 0,
                    'meds': -1,
                    'icd_level': 'chapter',
                    'atc_level': 'therapeutic',
                }
            }
        },
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.01
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler}
        },
    )
    LR_A100_SI100_OI100_BASIC_P1 = LR_BASIC._replace(
        description='',
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'diagnoses': 100,
                    'interventions': 0,
                    'meds': 100,
                    # 'icd_level': 'chapter',
                    # 'atc_level': 'therapeutic',
                }
            }
        },
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'history': {'processor': Binarizer},
            'basic': {'processor': StandardScaler}
        },
    )
    RF_A100_SI100_OI100_BASIC_P1 = LR_BASIC._replace(
        description='',
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'diagnoses': 100,
                    'interventions': 0,
                    'meds': 100,
                    # 'icd_level': 'chapter',
                    # 'atc_level': 'therapeutic',
                }
            }
        },
        model=RandomForestClassifier,
        model_kwargs={
            'class_weight': 'balanced',
            'n_jobs': -1,
            'n_estimators': 1000
        },
    )
    RF_A10_SI10_OI10_BASIC_P1 = RF_A100_SI100_OI100_BASIC_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV', 'OV'],
                    'diagnoses': 10,
                    'interventions': 0,
                    'meds': 10,
                    # 'icd_level': 'chapter',
                    # 'atc_level': 'therapeutic',
                }
            }
        },
    )
    RF_AC_SIC_OIC_BASIC_P1 = RF_A100_SI100_OI100_BASIC_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV', 'OV'],
                    'diagnoses': -1,
                    'interventions': 0,
                    'meds': -1,
                    'icd_level': 'chapter',
                    'atc_level': 'therapeutic',
                }
            }
        },
        model_kwargs={
            'class_weight': 'balanced',
            'n_jobs': -1,
            'n_estimators': 1000,
            'max_features': None
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
                    'diagnoses': 10,
                    'interventions': 0,
                    'meds': 0,
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
                    'diagnoses': 10,
                    'interventions': 0,
                    'meds': 0,
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
                    'diagnoses': 100,
                    'interventions': 0,
                    'meds': 0,
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
                    'diagnoses': 100,
                    'interventions': 0,
                    'meds': 0,
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
                    'diagnoses': 1000,
                    'interventions': 0,
                    'meds': 0,
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
                    'diagnoses': 1000,
                    'interventions': 0,
                    'meds': 0,
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
                    'diagnoses': 0,
                    'interventions': 10,
                    'meds': 0,
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
                    'diagnoses': 0,
                    'interventions': 10,
                    'meds': 0,
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
                    'diagnoses': 0,
                    'interventions': 100,
                    'meds': 0,
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
                    'diagnoses': 0,
                    'interventions': 100,
                    'meds': 0,
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
                    'diagnoses': 0,
                    'interventions': 1000,
                    'meds': 0,
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
                    'diagnoses': 0,
                    'interventions': 1000,
                    'meds': 0,
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
                    'diagnoses': 10,
                    'interventions': 0,
                    'meds': 0,
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
                    'diagnoses': 10,
                    'interventions': 0,
                    'meds': 0,
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
                    'diagnoses': 100,
                    'interventions': 0,
                    'meds': 0,
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
                    'diagnoses': 100,
                    'interventions': 0,
                    'meds': 0,
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
                    'diagnoses': 1000,
                    'interventions': 0,
                    'meds': 0,
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
                    'diagnoses': 1000,
                    'interventions': 0,
                    'meds': 0,
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
                    'diagnoses': 0,
                    'interventions': 10,
                    'meds': 0,
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
                    'diagnoses': 0,
                    'interventions': 10,
                    'meds': 0,
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
                    'diagnoses': 0,
                    'interventions': 100,
                    'meds': 0,
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
                    'diagnoses': 0,
                    'interventions': 100,
                    'meds': 0,
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
                    'diagnoses': 0,
                    'interventions': 1000,
                    'meds': 0,
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
                    'diagnoses': 0,
                    'interventions': 1000,
                    'meds': 0,
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
                    'diagnoses': 0,
                    'interventions': 0,
                    'meds': 10,
                }
            }
        },
    )

    LR_A10_BASIC_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'basic': ['age', 'sex'],
                'history': {
                    'intervals': {'periods': 1},
                    'diagnoses': 0,
                    'interventions': 0,
                    'meds': 10,
                }
            }
            # 'features': {
            #     'intervals': {'periods': 1},
            #     'diagnoses': 0,
            #     'interventions': 0,
            #     'meds': 10,
            # }
        },
    )
    LR_A10_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'diagnoses': 0,
                    'interventions': 0,
                    'meds': 10,
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
                    'diagnoses': 0,
                    'interventions': 0,
                    'meds': 100,
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
                    'diagnoses': 0,
                    'interventions': 0,
                    'meds': 100,
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
                    'diagnoses': 0,
                    'interventions': 0,
                    'meds': 1000,
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
                    'diagnoses': 0,
                    'interventions': 0,
                    'meds': 1000,
                }
            }
        },
    )

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
                    'diagnoses': -1,
                    'icd_level': 'chapter',
                    'interventions': 0,
                    'meds': 0,
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
    LR_OIC_P1 = LR_SIC_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['OV'],
                    'diagnoses': -1,
                    'icd_level': 'chapter',
                    'interventions': 0,
                    'meds': 0,
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
                    'diagnoses': 0,
                    'atc_level': 'therapeutic',
                    'interventions': 0,
                    'meds': -1,
                }
            }
        },
    )
    LR_AC_P1_MALE = LR_SIC_P1._replace(
        description='What if we try to predict sex?',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'diagnoses': 0,
                    'atc_level': 'therapeutic',
                    'interventions': 0,
                    'meds': -1,
                }
            },
            'labels': {'outcome': 'male'}
        },
    )
    LR_AC_SC_P1_MALE = LR_SIC_P1._replace(
        description='What if we try to predict sex?',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV'],
                    'diagnoses': -1,
                    'icd_level': 'chapter',
                    'atc_level': 'therapeutic',
                    'interventions': 0,
                    'meds': -1,
                }
            },
            'labels': {'outcome': 'male'}
        },
    )
    LR_A10_P1_MALE = LR_SIC_P1._replace(
        description='What if we try to predict sex?',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'diagnoses': 0,
                    # 'atc_level': 'therapeutic',
                    'interventions': 0,
                    'meds': 10,
                }
            },
            'labels': {'outcome': 'male'}
        },
    )
    LR_A100_P1_MALE = LR_SIC_P1._replace(
        description='What if we try to predict sex?',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'diagnoses': 0,
                    # 'atc_level': 'therapeutic',
                    'interventions': 0,
                    'meds': 100,
                }
            },
            'labels': {'outcome': 'male'}
        },
    )

    LR_AC_SC_P1_AGE = LR_SIC_P1._replace(
        description='What if we try to predict age?',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV'],
                    'diagnoses': -1,
                    'icd_level': 'chapter',
                    'atc_level': 'therapeutic',
                    'interventions': 0,
                    'meds': -1,
                }
            },
            'labels': {'outcome': 'age'}
        },
        scoring=r2_score,
        metrics=['r2', 'mae'],
        model=LinearRegression,
        model_kwargs={
        },
    )

    LR_A100_SI100_OI100_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 1},
                    'sources': ['SV', 'OV'],
                    'diagnoses': 100,
                    'interventions': 0,
                    'meds': 100,
                }
            }
        },
    )

    FOO = LR_SI10_P1._replace(
        description="This one overfits and sklearn raises a warning about "
                    "preprocessing the data. Let's see if I can fix that.",
        extractor_kwargs={
            'features': {
                'history': {
                    'intervals': {'periods': 5},
                    'diagnoses': 0,
                    'interventions': 0,
                    'meds': 1000,
                }
            }
        },
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300,
            'C': 0.001
        },
        pre_processor_kwargs={
            'history': {
                'processor': 'Pipeline',
                'steps': [
                    ('Binarizer', Binarizer, {'threshold': 0.0}),
                    ('pca', PCA, {'n_components': 50, 'random_state': 42})
                ]
            }
        },
    )

    BAR = Experiment(
        description="Test to see if the LSTM works",
        model=simple_lstm,
        model_kwargs={
        },
        extractor=Ragged,
        extractor_kwargs={
            'features': {
                'meds': 10,
            }
        },
        batch_size=512,
        epochs=50,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 3e-3}
        },
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43,
        },
        scoring=roc_auc_score,
        metrics=['accuracy', 'auc'],
        building_model_requires_development_data=True,
    )

    BAZ = Experiment(
        description="Test to see if the simple ffnn works",
        model=simple_ffnn,
        model_kwargs={
            'sizes': [10],
        },
        extractor=Ragged,
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 1},
                'diagnoses': 0,
                'interventions': 0,
                'meds': 10,
            }
        },
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43,
        },
        batch_size=32,
        epochs=50,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-3}
        },
        scoring=roc_auc_score,
        metrics=['accuracy', 'auc'],
        building_model_requires_development_data=True,
    )


class BadOutcomes(Experiment, Enum):
    LR_BASIC = Experiment(
        description='Predicting I200, I21, I22, I26, I71 or death',
        model=LogisticRegression,
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300
        },
        extractor=Flat,
        extractor_kwargs={
            'features': {'basic': ['age', 'sex']},
            'labels': {'outcome': 'BAD'}
        },
        # pre_processor=sklearn_process,
        # pre_processor_kwargs={
        #     'history': {'processor': Binarizer},
        # },
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43,
        },
        scoring=roc_auc_score,
        metrics=['accuracy', 'auc'],
    )
