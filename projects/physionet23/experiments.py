from enum import Enum

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
# from keras.optimizers.schedules.learning_rate_schedule import
# ExponentialDecay

from mim.experiments.experiments import Experiment
from mim.experiments.extractor import sklearn_process
from projects.physionet23.evaluate_model import new_challenge_score
from projects.physionet23.extractor import ICARE
from projects.physionet23.models import mlp, lstm


class Physionet23(Experiment, Enum):
    LR_BASIC = Experiment(
        description='Predict primary outcome using only the basic patient '
                    'features.',
        model=LogisticRegression,
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300
        },
        extractor=ICARE,
        extractor_kwargs={
            'features': {'patient_features': True},
            'labels': {'outcome': True, 'cpc': False}
        },
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 0.8,
            'random_state': 999
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'patient_features': {'processor': StandardScaler}
        },
        scoring=new_challenge_score,
        metrics=['accuracy', 'auc']
    )
    RF_BWM_LAST = Experiment(
        description='Random forest using mean brain-waves from the last EEG.',
        model=RandomForestClassifier,
        model_kwargs={
            'n_estimators': 1000,
            'n_jobs': -1,
            'random_state': 42
        },
        extractor=ICARE,
        extractor_kwargs={
            'features': {
                'patient_features': True,
                'brain_waves': {
                    'which': 'last',  # Use only the last EEG
                    'mean': True,
                    'alpha': True,
                    'beta': True,
                    'delta': True,
                    'theta': True
                }
            },
            'labels': {'outcome': True, 'cpc': False}
        },
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 0.8,
            'random_state': 999
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'patient_features': {'processor': StandardScaler}
        },
        scoring=new_challenge_score,
        metrics=['accuracy', 'auc']
    )
    MLP1_BWM_ALL = Experiment(
        description='MLP with single hidden layer, using mean brain-waves '
                    'from all 72 EEGs.',
        model=mlp,
        model_kwargs={
            'patient_mlp_kwargs': None,
            'bw_mlp_kwargs': None,
            'final_mlp_kwargs': {
                'sizes': [500],
                'dropout': 0.5,
                'batch_norm': False,
                'regularizer': {
                    'activity': 0.3,
                    'bias': 0.3,
                    'kernel': 0.3
                }
            }
        },
        extractor=ICARE,
        extractor_kwargs={
            'features': {
                'patient_features': True,
                'eeg_features': {
                    'brain_waves': {
                        'which': 'all',
                        'mean': True,
                        'alpha': True,
                        'beta': True,
                        'delta': True,
                        'theta': True
                    }
                },
            },
            'labels': {'outcome': True, 'cpc': False}
        },
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 0.8,
            'random_state': 999
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'patient_features': {'processor': StandardScaler}
        },
        building_model_requires_development_data=True,
        batch_size=128,
        epochs=1000,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.00003}
        },
        scoring=new_challenge_score,
        metrics=['accuracy', 'auc']
    )
    LSTM1_BWM_ALL = Experiment(
        description='Simple LSTM using mean brain-wave features from all 72 '
                    'hours.',
        model=lstm,
        model_kwargs={
            'patient_mlp_kwargs': None,
            'lstm_kwargs': {
                'units': 50,
                'dropout': 0.0
            },
            'final_mlp_kwargs': {
                'sizes': [10],
                'dropout': 0.1,
                'batch_norm': False,
                'regularizer': {
                    'activity': 0.2,
                    'bias': 0.2,
                    'kernel': 0.2
                }
            }
        },
        extractor=ICARE,
        extractor_kwargs={
            'features': {
                'patient_features': True,
                'eeg_features': {
                    'lstm': True,
                    'brain_waves': {
                        'which': 'all',
                        'mean': True,
                        'alpha': True,
                        'beta': True,
                        'delta': True,
                        'theta': True
                    },
                },
            },
            'labels': {'outcome': True, 'cpc': False}
        },
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 0.8,
            'random_state': 999
        },
        pre_processor=sklearn_process,
        pre_processor_kwargs={
            'patient_features': {'processor': StandardScaler}
        },
        building_model_requires_development_data=True,
        batch_size=128,
        epochs=1000,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.001}
            # 'kwargs': {
            #     'learning_rate': {
            #         'scheduler': ExponentialDecay,
            #         'scheduler_kwargs': {
            #             'initial_learning_rate': 0.003,
            #             'decay_steps': 4*400,
            #             'decay_rate': 0.1,
            #         }
            #     }
            # }
        },
        save_learning_rate=True,
        scoring=new_challenge_score,
        metrics=['accuracy', 'auc']
    )
