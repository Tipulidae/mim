from enum import Enum

from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import ShuffleSplit
from tensorflow.keras.optimizers import Adam

from mim.experiments.experiments import Experiment
from mim.experiments.hyper_experiments import HyperExperiment
from mim.experiments import hyper_parameter as hp
from mim.experiments.search_strategies import RandomSearch
from mim.extractors.ptbxl import PTBXL
from mim.models.simple_nn import ptbxl_cnn


class ptbxl(Experiment, Enum):
    CNN1_12L_SEX = Experiment(
        description="First attempt to train model predicting sex, using all "
                    "12 leads.",
        model=ptbxl_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': False,
                'num_layers': 3,
                'dropout': 0.3,
                'filter_first': 32,
                'filter_last': 16,
                'kernel_first': 11,
                'kernel_last': 5,
            },
            'ffnn_kwargs': {
                'sizes': [100, 10],
                'dropouts': [0.3, 0.3],
                'batch_norms': [False, False],
                'activity_regularizer': 0.01,
                'kernel_regularizer': 0.01,
                'bias_regularizer': 0.01,
            },
        },
        extractor=PTBXL,
        extractor_kwargs={
            'labels': {'sex': True},
            'features': {'leads': 12, 'resolution': 'high'}
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.001,
            }
        },
        epochs=200,
        batch_size=64,
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2/3,
            'random_state': 43
        },
        scoring=roc_auc_score,
        metrics=['accuracy', 'auc']
    )

    CNN1_12L_AGE = Experiment(
        description="First attempt to train model predicting age, using all "
                    "12 leads.",
        model=ptbxl_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': False,
                'num_layers': 3,
                'dropout': 0.3,
                'filter_first': 32,
                'filter_last': 16,
                'kernel_first': 11,
                'kernel_last': 5,
            },
            'ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.3],
                'batch_norms': [False],
                'activity_regularizer': 0.01,
                'kernel_regularizer': 0.01,
                'bias_regularizer': 0.01,
            },
        },
        extractor=PTBXL,
        extractor_kwargs={
            'labels': {'age': True},
            'features': {'leads': 12, 'resolution': 'high'}
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.003,
            }
        },
        epochs=200,
        batch_size=64,
        building_model_requires_development_data=True,
        loss='mean_absolute_error',
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2/3,
            'random_state': 43
        },
        scoring=r2_score,
        metrics=['mae', 'mse']
    )

    CNN1_12L_HEIGHT = Experiment(
        description="First attempt to train model predicting height, using "
                    "all 12 leads.",
        model=ptbxl_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': False,
                'num_layers': 3,
                'dropout': 0.3,
                'filter_first': 32,
                'filter_last': 16,
                'kernel_first': 11,
                'kernel_last': 5,
            },
            'ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.3],
                'batch_norms': [False],
                'activity_regularizer': 0.01,
                'kernel_regularizer': 0.01,
                'bias_regularizer': 0.01,
            },
        },
        extractor=PTBXL,
        extractor_kwargs={
            'labels': {'height': True},
            'features': {'leads': 12, 'resolution': 'high'}
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.001,
            }
        },
        epochs=200,
        batch_size=64,
        building_model_requires_development_data=True,
        loss='mean_absolute_error',
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2/3,
            'random_state': 43
        },
        scoring=r2_score,
        metrics=['mae', 'mse']
    )

    CNN1_12L_WEIGHT = Experiment(
        description="First attempt to train model predicting weight, using "
                    "all 12 leads.",
        model=ptbxl_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': False,
                'num_layers': 3,
                'dropout': 0.3,
                'filter_first': 32,
                'filter_last': 16,
                'kernel_first': 11,
                'kernel_last': 5,
            },
            'ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.3],
                'batch_norms': [False],
                'activity_regularizer': 0.01,
                'kernel_regularizer': 0.01,
                'bias_regularizer': 0.01,
            },
        },
        extractor=PTBXL,
        extractor_kwargs={
            'labels': {'weight': True},
            'features': {'leads': 12, 'resolution': 'high'}
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.001,
            }
        },
        epochs=200,
        batch_size=64,
        building_model_requires_development_data=True,
        loss='mean_absolute_error',
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2/3,
            'random_state': 43
        },
        scoring=r2_score,
        metrics=['mae', 'mse']
    )

    CNN1_12L_SEX_AGE_HEIGHT_WEIGHT = Experiment(
        description="First attempt to train model predicting all outcomes at "
                    "the same time, using 12 leads.",
        model=ptbxl_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': False,
                'num_layers': 3,
                'dropout': 0.3,
                'filter_first': 32,
                'filter_last': 16,
                'kernel_first': 11,
                'kernel_last': 5,
            },
            'ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.3],
                'batch_norms': [False],
                'activity_regularizer': 0.01,
                'kernel_regularizer': 0.01,
                'bias_regularizer': 0.01,
            },
        },
        extractor=PTBXL,
        extractor_kwargs={
            'labels': {
                'sex': True,
                'age': True,
                'height': True,
                'weight': True
            },
            'features': {'leads': 12, 'resolution': 'high'}
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.003,
            }
        },
        epochs=500,
        batch_size=64,
        building_model_requires_development_data=True,
        loss={
            'sex': 'binary_crossentropy',
            'age': 'mean_absolute_error',
            'weight': 'mean_absolute_error',
            'height': 'mean_absolute_error',
        },
        loss_weights=[25.0, 1.0, 1.0, 1.0],
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2/3,
            'random_state': 43
        },
        scoring=None,
        metrics={'sex': ['acc', 'auc']}
    )

    CNN1_1L_SEX = CNN1_12L_SEX._replace(
        description="First attempt to train model predicting sex, using "
                    "only lead I.",
        extractor_kwargs={
            'labels': {'sex': True},
            'features': {'leads': 1, 'resolution': 'high'}
        },
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': False,
                'num_layers': 3,
                'dropout': 0.1,
                'filter_first': 16,
                'filter_last': 32,
                'kernel_first': 41,
                'kernel_last': 5,
                'batch_norm': True,
                # 'weight_decay': 0.01
            },
            'ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.2],
                'batch_norms': [False],
                # 'activity_regularizer': 0.01,
                # 'kernel_regularizer': 0.001,
                # 'bias_regularizer': 0.001,
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.0003,
            }
        },
        epochs=200,
    )

    CNN1_1L_AGE = CNN1_12L_AGE._replace(
        description="First attempt to train model predicting age, using "
                    "only lead I.",
        extractor_kwargs={
            'labels': {'age': True},
            'features': {'leads': 1, 'resolution': 'high'}
        },
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': False,
                'num_layers': 3,
                'dropout': 0.1,
                'filter_first': 16,
                'filter_last': 32,
                'kernel_first': 41,
                'kernel_last': 5,
                'batch_norm': True,
                # 'weight_decay': 0.01
            },
            'ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.2],
                'batch_norms': [False],
                # 'activity_regularizer': 0.01,
                # 'kernel_regularizer': 0.001,
                # 'bias_regularizer': 0.001,
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.0003,
            }
        },
        epochs=200,
        loss='mean_absolute_error',
        scoring=r2_score,
        metrics=['mae', 'mse']
    )

    CNN1_1L_HEIGHT = CNN1_12L_HEIGHT._replace(
        description="First attempt to train model predicting height, using "
                    "only lead I.",
        extractor_kwargs={
            'labels': {'height': True},
            'features': {'leads': 1, 'resolution': 'high'}
        },
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': False,
                'num_layers': 3,
                'dropout': 0.1,
                'filter_first': 16,
                'filter_last': 32,
                'kernel_first': 41,
                'kernel_last': 5,
                'batch_norm': True,
                # 'weight_decay': 0.01
            },
            'ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.2],
                'batch_norms': [False],
                # 'activity_regularizer': 0.01,
                # 'kernel_regularizer': 0.001,
                # 'bias_regularizer': 0.001,
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.0003,
            }
        },
        epochs=200,
        loss='mean_absolute_error',
        scoring=r2_score,
        metrics=['mae', 'mse']
    )

    CNN1_1L_WEIGHT = CNN1_12L_WEIGHT._replace(
        description="First attempt to train model predicting weight, using "
                    "only lead I.",
        extractor_kwargs={
            'labels': {'weight': True},
            'features': {'leads': 1, 'resolution': 'high'}
        },
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': False,
                'num_layers': 3,
                'dropout': 0.1,
                'filter_first': 16,
                'filter_last': 32,
                'kernel_first': 41,
                'kernel_last': 5,
                'batch_norm': True,
                # 'weight_decay': 0.01
            },
            'ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.2],
                'batch_norms': [False],
                # 'activity_regularizer': 0.01,
                # 'kernel_regularizer': 0.001,
                # 'bias_regularizer': 0.001,
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.0003,
            }
        },
        epochs=200,
        loss='mean_absolute_error',
        scoring=r2_score,
        metrics=['mae', 'mse']
    )

    CNN1_1L_SEX_AGE_HEIGHT_WEIGHT = CNN1_12L_SEX_AGE_HEIGHT_WEIGHT._replace(
        description="First attempt to train model predicting all outcomes at "
                    "the same time, using only lead I.",
        model=ptbxl_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': False,
                'num_layers': 4,
                'dropout': 0.2,
                'filter_first': 32,
                'filter_last': 16,
                'kernel_first': 41,
                'kernel_last': 5,
            },
            'ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.1],
                'batch_norms': [False],
                'activity_regularizer': 0.01,
                'kernel_regularizer': 0.01,
                'bias_regularizer': 0.01,
            },
        },
        extractor=PTBXL,
        extractor_kwargs={
            'labels': {
                'sex': True,
                'age': True,
                'height': True,
                'weight': True
            },
            'features': {'leads': 1, 'resolution': 'high'}
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.003,
            }
        },
        epochs=500,
        batch_size=64,
        building_model_requires_development_data=True,
        loss={
            'sex': 'binary_crossentropy',
            'age': 'mean_absolute_error',
            'weight': 'mean_absolute_error',
            'height': 'mean_absolute_error',
        },
        loss_weights=[25.0, 1.0, 1.0, 1.0],
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2/3,
            'random_state': 43
        },
        scoring=None,
        metrics={'sex': ['acc', 'auc']}
    )

    CNN2_1L_SEX_AGE_HEIGHT_WEIGHT = CNN1_1L_SEX_AGE_HEIGHT_WEIGHT._replace(
        description='Try adding separate dense layers for each output.',
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': False,
                'num_layers': 4,
                'dropout': 0.2,
                'filter_first': 32,
                'filter_last': 16,
                'kernel_first': 41,
                'kernel_last': 5,
            },
            'ffnn_kwargs': {
                'sizes': [50],
                'dropouts': [0.1],
            },
            'final_ffnn_kwargs': {
                'age': {'sizes': [10]},
                'weight': {'sizes': [10]},
                'sex': {
                    'sizes': [10],
                    'default_regularizer': 0.003,
                    'default_dropout': 0.1
                },
                'height': {
                    'sizes': [20],
                    # 'default_regularizer': 0.003,
                    # 'default_dropout': 0.3
                },
            }
        },
        epochs=200,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.001,
            }
        },
    )


class HyperPTBXL(HyperExperiment, Enum):
    RS_1L_ALL = HyperExperiment(
        template=ptbxl.CNN1_1L_SEX_AGE_HEIGHT_WEIGHT._replace(
            description="",
            model_kwargs={
                'cnn_kwargs': {
                    'down_sample': False,
                    'num_layers': hp.Choice([2, 3, 4]),
                    'dropout': hp.Choice([0.0, 0.1, 0.2, 0.3]),
                    'filter_first': hp.Choice([16, 32, 48]),
                    'filter_last': hp.Choice([16, 32, 48]),
                    'kernel_first': hp.Choice([11, 21, 31, 41, 51]),
                    'kernel_last': hp.Choice([5, 7, 11]),
                    'weight_decay': hp.Choice([0.03, 0.01, 0.003, 0.001, 0.0])
                },
                'ffnn_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [10, 25, 50, 100], k=num_layers, ascending=False
                        ),
                        'default_dropout': hp.Choice([0.0, 0.1, 0.2, 0.3]),
                        'default_regularizer': hp.Choice(
                            [0.03, 0.01, 0.003, 0.001, 0.0]
                        )
                    } for num_layers in [1, 2]
                ]),
                'final_ffnn_kwargs': hp.Choice([
                    None,
                    {
                        'age': {
                            'sizes': hp.Choice([[5], [10], [20]]),
                            'default_regularizer': hp.Choice(
                                [0.01, 0.003, 0.001, 0.0]),
                            'default_dropout': hp.Choice([0.0, 0.1, 0.2, 0.3])
                        },
                        'weight': {
                            'sizes': hp.Choice([[5], [10], [20]]),
                            'default_regularizer': hp.Choice(
                                [0.01, 0.003, 0.001, 0.0]),
                            'default_dropout': hp.Choice([0.0, 0.1, 0.2, 0.3])
                        },
                        'sex': {
                            'sizes': hp.Choice([[5], [10], [20]]),
                            'default_regularizer': hp.Choice(
                                [0.01, 0.003, 0.001, 0.0]),
                            'default_dropout': hp.Choice([0.0, 0.1, 0.2, 0.3])
                        },
                        'height': {
                            'sizes': hp.Choice([[5], [10], [20]]),
                            'default_regularizer': hp.Choice(
                                [0.01, 0.003, 0.001, 0.0]),
                            'default_dropout': hp.Choice([0.0, 0.1, 0.2, 0.3])
                        },
                    }
                ]),
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice([0.003, 0.001, 0.0003, 0.0001]),
                }
            },
            epochs=300,
            batch_size=64,
            metrics={'sex': ['acc', 'auc']},
            ignore_callbacks=True
        ),
        random_seed=42,
        strategy=RandomSearch,
        strategy_kwargs={'iterations': 1000},
    )
