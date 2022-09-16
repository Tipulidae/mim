from enum import Enum

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import PCA

from mim.experiments.experiments import Experiment
from mim.experiments.extractor import sklearn_process
from projects.patient_history.extractor import Flat


class PatientHistory(Experiment, Enum):
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
                'intervals': {'periods': 1},
                'sources': ['SV'],
                'diagnoses': 10,
                'interventions': 0,
                'meds': 0,
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
                'intervals': {'periods': 5},
                'sources': ['SV'],
                'diagnoses': 10,
                'interventions': 0,
                'meds': 0,
            }
        },
    )
    LR_SI100_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 1},
                'sources': ['SV'],
                'diagnoses': 100,
                'interventions': 0,
                'meds': 0,
            }
        },
    )
    LR_SI100_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 5},
                'sources': ['SV'],
                'diagnoses': 100,
                'interventions': 0,
                'meds': 0,
            }
        },
    )
    LR_SI1000_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 1},
                'sources': ['SV'],
                'diagnoses': 1000,
                'interventions': 0,
                'meds': 0,
            }
        },
    )
    LR_SI1000_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 5},
                'sources': ['SV'],
                'diagnoses': 1000,
                'interventions': 0,
                'meds': 0,
            }
        },
    )

    LR_SK10_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 1},
                'sources': ['SV'],
                'diagnoses': 0,
                'interventions': 10,
                'meds': 0,
            }
        },
    )
    LR_SK10_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 5},
                'sources': ['SV'],
                'diagnoses': 0,
                'interventions': 10,
                'meds': 0,
            }
        },
    )
    LR_SK100_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 1},
                'sources': ['SV'],
                'diagnoses': 0,
                'interventions': 100,
                'meds': 0,
            }
        },
    )
    LR_SK100_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 5},
                'sources': ['SV'],
                'diagnoses': 0,
                'interventions': 100,
                'meds': 0,
            }
        },
    )
    LR_SK1000_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 1},
                'sources': ['SV'],
                'diagnoses': 0,
                'interventions': 1000,
                'meds': 0,
            }
        },
    )
    LR_SK1000_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 5},
                'sources': ['SV'],
                'diagnoses': 0,
                'interventions': 1000,
                'meds': 0,
            }
        },
    )

    LR_OI10_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 1},
                'sources': ['OV'],
                'diagnoses': 10,
                'interventions': 0,
                'meds': 0,
            }
        },
    )
    LR_OI10_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 5},
                'sources': ['OV'],
                'diagnoses': 10,
                'interventions': 0,
                'meds': 0,
            }
        },
    )
    LR_OI100_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 1},
                'sources': ['OV'],
                'diagnoses': 100,
                'interventions': 0,
                'meds': 0,
            }
        },
    )
    LR_OI100_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 5},
                'sources': ['OV'],
                'diagnoses': 100,
                'interventions': 0,
                'meds': 0,
            }
        },
    )
    LR_OI1000_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 1},
                'sources': ['OV'],
                'diagnoses': 1000,
                'interventions': 0,
                'meds': 0,
            }
        },
    )
    LR_OI1000_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 5},
                'sources': ['OV'],
                'diagnoses': 1000,
                'interventions': 0,
                'meds': 0,
            }
        },
    )

    LR_OK10_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 1},
                'sources': ['OV'],
                'diagnoses': 0,
                'interventions': 10,
                'meds': 0,
            }
        },
    )
    LR_OK10_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 5},
                'sources': ['OV'],
                'diagnoses': 0,
                'interventions': 10,
                'meds': 0,
            }
        },
    )
    LR_OK100_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 1},
                'sources': ['OV'],
                'diagnoses': 0,
                'interventions': 100,
                'meds': 0,
            }
        },
    )
    LR_OK100_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 5},
                'sources': ['OV'],
                'diagnoses': 0,
                'interventions': 100,
                'meds': 0,
            }
        },
    )
    LR_OK1000_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 1},
                'sources': ['OV'],
                'diagnoses': 0,
                'interventions': 1000,
                'meds': 0,
            }
        },
    )
    LR_OK1000_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 5},
                'sources': ['OV'],
                'diagnoses': 0,
                'interventions': 1000,
                'meds': 0,
            }
        },
    )

    LR_A10_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 1},
                'diagnoses': 0,
                'interventions': 0,
                'meds': 10,
            }
        },
    )
    LR_A10_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 5},
                'diagnoses': 0,
                'interventions': 0,
                'meds': 10,
            }
        },
    )
    LR_A100_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 1},
                'diagnoses': 0,
                'interventions': 0,
                'meds': 100,
            }
        },
    )
    LR_A100_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 5},
                'diagnoses': 0,
                'interventions': 0,
                'meds': 100,
            }
        },
    )
    LR_A1000_P1 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 1},
                'diagnoses': 0,
                'interventions': 0,
                'meds': 1000,
            }
        },
    )
    LR_A1000_P5 = LR_SI10_P1._replace(
        description='',
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 5},
                'diagnoses': 0,
                'interventions': 0,
                'meds': 1000,
            }
        },
    )

    FOO = LR_SI10_P1._replace(
        description="This one overfits and sklearn raises a warning about "
                    "preprocessing the data. Let's see if I can fix that.",
        extractor_kwargs={
            'features': {
                'intervals': {'periods': 5},
                'diagnoses': 0,
                'interventions': 0,
                'meds': 1000,
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
