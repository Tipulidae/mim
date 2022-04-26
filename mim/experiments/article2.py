from enum import Enum

from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA

from mim.experiments.experiments import Experiment
from mim.extractors import sk1718
from mim.extractors.extractor import sklearn_process
from mim.models.article2 import simple_ffnn, simple_lstm


class SK1718(Experiment, Enum):
    LR_LAB_ATC_ICD = Experiment(
        description='Baseline logistic regresion, using lab-values, atc '
                    'and icd-codes.',
        model=LogisticRegression,
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300
        },
        extractor=sk1718.Flat,
        extractor_kwargs={
            'features': {
                'lab_values': {},
                'comorbidities': {},
                'medicine': {}
            }
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler},
            'lab_values': {
                'processor': PowerTransformer,
                'method': 'box-cox'
            },
            'comorbidities': {
                'processor': PCA,
                'n_components': 33,
                'random_state': 42,
                'allow_categorical': True
            },
            'medicine': {
                'processor': PCA,
                'n_components': 26,
                'random_state': 42,
                'allow_categorical': True
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
    LR_LAB_ATC = LR_LAB_ATC_ICD._replace(
        description='Baseline logistic regresion, using lab-values and atc '
                    'codes.',
        extractor_kwargs={
            'features': {
                'lab_values': {},
                'medicine': {}
            }
        },
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler},
            'lab_values': {
                'processor': PowerTransformer,
                'method': 'box-cox'
            },
            'medicine': {
                'processor': PCA,
                'n_components': 26,
                'random_state': 42,
                'allow_categorical': True
            }
        },
    )
    LR_LAB_ICD = LR_LAB_ATC_ICD._replace(
        description='Baseline logistic regresion, using lab-values, and '
                    'icd-codes.',
        extractor_kwargs={
            'features': {
                'lab_values': {},
                'comorbidities': {},
            }
        },
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler},
            'lab_values': {
                'processor': PowerTransformer,
                'method': 'box-cox'
            },
            'comorbidities': {
                'processor': PCA,
                'n_components': 33,
                'random_state': 42,
                'allow_categorical': True
            },
        },
    )
    LR_LAB = LR_LAB_ATC_ICD._replace(
        description='Baseline logistic regresion, using only lab-values.',
        extractor_kwargs={
            'features': {
                'lab_values': {},
            }
        },
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler},
            'lab_values': {
                'processor': PowerTransformer,
                'method': 'box-cox'
            },
        },
    )
    LR_ATC_ICD = LR_LAB_ATC_ICD._replace(
        description='Baseline logistic regresion, using atc and icd codes.',
        extractor_kwargs={
            'features': {
                'comorbidities': {},
                'medicine': {}
            }
        },
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler},
            'comorbidities': {
                'processor': PCA,
                'n_components': 30,
                'random_state': 42,
                'allow_categorical': True
            },
            'medicine': {
                'processor': PCA,
                'n_components': 49,
                'random_state': 42,
                'allow_categorical': True
            }
        },
    )
    LR_ATC = LR_LAB_ATC_ICD._replace(
        description='Baseline logistic regresion, using only atc-codes.',
        extractor_kwargs={
            'features': {
                'medicine': {}
            }
        },
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler},
            'medicine': {
                'processor': PCA,
                'n_components': 49,
                'random_state': 42,
                'allow_categorical': True
            }
        },
    )
    LR_ICD = LR_LAB_ATC_ICD._replace(
        description='Baseline logistic regresion, using only icd-codes.',
        extractor_kwargs={
            'features': {
                'comorbidities': {},
            }
        },
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler},
            'comorbidities': {
                'processor': PCA,
                'n_components': 30,
                'random_state': 42,
                'allow_categorical': True
            },
        },
    )
    LR_BASIC = LR_LAB_ATC_ICD._replace(
        description='Baseline logistic regresion, using only age and sex.',
        extractor_kwargs={
            'features': {}
        },
        pre_processor_kwargs={
            'basic': {'processor': StandardScaler},
        },
    )

    QUICK_DIRTY = Experiment(
        description='Quick and dirty experiment with data0 and small network.',
        model=simple_ffnn,
        model_kwargs={
            'sizes': [100, 10],
            'dropouts': [0.5, 0.5],
            'batch_norms': [False, False],
            'activity_regularizer': 0.01,
            'kernel_regularizer': 0.01,
            'bias_regularizer': 0.01,
        },
        extractor=sk1718.Flat,
        extractor_kwargs={
            'features': ['medicine', 'comorbidities']
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.00001,
            }
        },
        epochs=200,
        batch_size=64,
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2/3,
            'random_state': 43,
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc'],
    )
    QUICK_DIRTY_D1 = Experiment(
        description='Quick and dirty experiment with data0 + data1 and small '
                    'network.',
        model=simple_ffnn,
        model_kwargs={
            'sizes': [200, 100, 10],
            'dropouts': [0.5, 0.5, 0.5],
            'batch_norms': [False, False, False],
            'activity_regularizer': 0.01,
            'kernel_regularizer': 0.01,
            'bias_regularizer': 0.01,
        },
        extractor=sk1718.Flat,
        extractor_kwargs={
            'features': {
                'medicine': True,
                'comorbidities': True,
                'lab_values': ['data1']
            }
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.00001,
            }
        },
        epochs=10,
        batch_size=64,
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2/3,
            'random_state': 43,
        },
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc'],
    )
    QUICK_DIRTY_D2 = QUICK_DIRTY_D1._replace(
        description='Quick and dirty experiment with data0 + data1 + data2 '
                    'and small network.',
        extractor_kwargs={
            'features': {
                'medicine': True,
                'comorbidities': True,
                'lab_values': ['data1', 'data2']
            }
        },
    )

    LSTM_TEST = Experiment(
        description='Simple test with LSTM',
        model=simple_lstm,
        extractor=sk1718.Flat,
        extractor_kwargs={},
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43,
        },
        scoring=roc_auc_score,
        metrics=['accuracy', 'auc'],
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.0003,
            }
        },
        epochs=200,
        batch_size=64,
    )
