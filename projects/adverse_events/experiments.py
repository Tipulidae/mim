from enum import Enum

from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from mim.experiments import hyper_parameter as hp
from mim.experiments.experiments import Experiment
from mim.experiments.hyper_experiments import HyperExperiment
from mim.experiments.search_strategies import RandomSearch
from projects.adverse_events.extractor import SelfControl, PredictionTasks
from projects.adverse_events.models import autoencoder_functional, mlp_prediction

"""
class AutoEncoder(Experiment, Enum):

    AE_BASIC = Experiment(
        description='Basic autoencoder with 30 latent dimensions.',
        model=autoencoder_functional,
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [218, 81, 30, 81, 218],
                # 'dropout': [0.2, 0.5, 0.5],
                # 'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
        extractor=SelfControl,
        extractor_kwargs={},
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.001,
            }
        },
        epochs=200,
        batch_size=32,
        building_model_requires_development_data=True,
        loss={'Med': 'mse', 'Age': 'mse', 'Gender': 'categorical_crossentropy'},
        loss_weights={'Med': 0.995, 'Age': 0.004, 'Gender': 0.001},
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 10,
            'train_size': 2 / 3,
            'random_state': 43
        },
        scoring=None,
        reconstruction_error=True,
        metrics=['accuracy']
    )
    AE_LATENT_1 = AE_BASIC._replace(
        description='AE with 1 in latent dimension.',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [70, 8, 1, 8, 70],
                # 'dropout': [0.2, 0.5, 0.5],
                # 'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
    )
    AE_LATENT_5 = AE_BASIC._replace(
        description='AE with 5 in latent dimension.',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [121, 25, 5, 25, 121],
                # 'dropout': [0.2, 0.5, 0.5],
                # 'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
    )
    AE_LATENT_10 = AE_BASIC._replace(
        description='AE with 10 in latent dimension.',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [152, 39, 10, 39, 152],
                # 'dropout': [0.2, 0.5, 0.5],
                # 'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
    )
    AE_LATENT_15 = AE_BASIC._replace(
        description='AE with 15 in latent dimension.',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [174, 51, 15, 51, 174],
                # 'dropout': [0.2, 0.5, 0.5],
                # 'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
    )
    AE_LATENT_20 = AE_BASIC._replace(
        description='AE with 20 in latent dimension.',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [192, 62, 20, 62, 192],
                # 'dropout': [0.2, 0.5, 0.5],
                # 'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
    )
    AE_LATENT_25 = AE_BASIC._replace(
        description='AE with 25 in latent dimension.',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [207, 72, 25, 72, 207],
                # 'dropout': [0.2, 0.5, 0.5],
                # 'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
    )
    AE_LATENT_30 = AE_BASIC._replace(
        description='AE with 30 in latent dimension.',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [220, 81, 30, 81, 220],
                # 'dropout': [0.2, 0.5, 0.5],
                # 'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
    )
    AE_LATENT_40 = AE_BASIC._replace(
        description='AE with 40 in latent dimension.',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [242, 98, 40, 98, 242],
                # 'dropout': [0.2, 0.5, 0.5],
                # 'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
    )
    AE_LATENT_50 = AE_BASIC._replace(
        description='AE with 50 in latent dimension.',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [260, 114, 50, 114, 260],
                # 'dropout': [0.2, 0.5, 0.5],
                # 'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
    )
    AE_LATENT_60 = AE_BASIC._replace(
        description='AE with 60 in latent dimension.',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [260, 114, 60, 114, 260],
                # 'dropout': [0.2, 0.5, 0.5],
                # 'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
    )
    AE_LATENT_100 = AE_BASIC._replace(
        description='AE with 100 in latent dimension.',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [328, 181, 100, 181, 328],
                # 'dropout': [0.2, 0.5, 0.5],
                # 'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
    )
    AE_LATENT_200 = AE_BASIC._replace(
        description='AE with 200 in latent dimension.',
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [413, 287, 200, 287, 413],
                # 'dropout': [0.2, 0.5, 0.5],
                # 'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
    )

    mlp_prediction_new = Experiment(
        description='Testing to predict the vaccination ',
        model=mlp_prediction,
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [400, 100, 10],
                'dropout': [0.2, 0.1, 0.1],
                # 'regularizer': [1e-3, 1e-4, 1e-4],
            },
        },
        extractor=PredictionTasks,
        extractor_kwargs={},
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.001,
            }
        },
        epochs=15,
        batch_size=512,
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 3,
            'train_size': 2 / 3,
            'random_state': 43
        },
        scoring=None,
        metrics=['accuracy']
    )
    mlp_prediction = Experiment(
        description='Testing to predict the vaccination ',
        model=mlp_prediction,
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [600, 200, 30],
                'dropout': [0.1, 0.1, 0.1],
                'regularizer': [1e-4, 1e-4, 1e-4],
            },
        },
        extractor=PredictionTasks,
        extractor_kwargs={},
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.001,
            }
        },
        epochs=15,
        batch_size=512,
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 35,
            'train_size': 9/10,
            'random_state': 43
        },
        scoring=None,
        metrics=['accuracy']
    )

    mlp_prediction_NO_U_ZV100 = mlp_prediction._replace(
        extractor_kwargs={
            'drop_name': ['U11', 'ZV100', 'Z25']
        }
    )
    mlp_prediction_NO_FAKE = mlp_prediction._replace(
        extractor_kwargs={
            'drop_name': ['U11', 'ZV100', 'Z25'],
        },
        cv_kwargs={
            'n_splits': 35,
            'train_size': 9 / 10,
            'random_state': 43
        },
    )
    mlp_prediction_simple_fake1 = mlp_prediction._replace(
        extractor_kwargs={
            'drop_name': ['U11', 'ZV100', 'Z25'],
            'simple_fake': [[0, 0.5, 0.75, 0.875], [0.95, 0.03, 0.01, 0.01], [0.75, 0.15, 0.08, 0.02]]
        },
        cv_kwargs={
            'n_splits': 35,
            'train_size': 9 / 10,
            'random_state': 43
        },
    )

    mlp_prediction_NO_HISTORY = mlp_prediction._replace(
        extractor_kwargs={
            'drop_name': ['A-B', 'C-D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S-T',
                          'U', 'V-Y', 'Z', 'W']
        }
    )

    random_forest_prediction = Experiment(
        description='Testing to predict the vaccination ',
        model=RandomForestClassifier,
        model_kwargs={
            'n_estimators': 1000,
            'random_state': 44,
            'n_jobs': -1,
            'verbose': 1
        },
        extractor=PredictionTasks,
        extractor_kwargs={},
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43
        },
        scoring=None,
        metrics=['accuracy']
    )
    xgboost_prediction = Experiment(
        description='Testing to predict the vaccination with XGBOOST',
        model=XGBClassifier,
        model_kwargs={
        },
        extractor=PredictionTasks,
        extractor_kwargs={},
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43
        },
        scoring=None,
        metrics=['accuracy']
    )
    xgboost_prediction_simple_fake1 = xgboost_prediction._replace(
        extractor_kwargs={
            'drop_name': ['U11', 'ZV100', 'Z25'],
            'simple_fake': [[0, 0.5, 0.75, 0.875], [0.95, 0.03, 0.01, 0.01], [0.75, 0.15, 0.08, 0.02]]
        },
        cv_kwargs={
            'n_splits': 35,
            'train_size': 9 / 10,
            'random_state': 43
        },
    )
    xgboost_prediction_no_fake = xgboost_prediction._replace(
        extractor_kwargs={
            'drop_name': ['U11', 'ZV100', 'Z25'],
        },
        cv_kwargs={
            'n_splits': 35,
            'train_size': 9 / 10,
            'random_state': 43
        },
    )
    xgboost_prediction_no_u_zv = xgboost_prediction._replace(
        extractor_kwargs={
            'drop_name': ['U00-U49', 'ZV100']
        }
    )
    LR_PREDICTION = Experiment(
        description='Basic logistic regression, to predict if data is before or after vaccination.',
        model=LogisticRegression,
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300
        },
        extractor=PredictionTasks,
        extractor_kwargs={},
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 9 / 10,
            'random_state': 43
        },
        metrics=['accuracy']
    )
"""


class Prediction(Experiment, Enum):
    mlp_prediction = Experiment(
        description='Testing to predict the vaccination ',
        model=mlp_prediction,
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [20, 10, 5],
                'dropout': [0.0, 0.0, 0.0],
                'regularizer': {
                    "bias": [0, 0, 0],
                    "activity": [0.0001, 0, 0],
                    "kernel": [0, 0, 0.0001]
                },
            },
        },
        extractor=PredictionTasks,
        extractor_kwargs={
                'drop_name': ['U11', 'ZV100', 'Z25'],
            },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.001,
            }
        },
        epochs=10,
        batch_size=512,
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 15,
            'train_size': 9 / 10,
            'random_state': 43
        },
        scoring=None,
        metrics=['accuracy']
    )

    xp_86_new = Experiment(
        description='Making an experiemnt to continue for xp_86 for rs_layer10',
        model=mlp_prediction,
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [50],
                'dropout': [0.3],
                'regularizer': [1e-4],
            },
        },
        extractor=PredictionTasks,
        extractor_kwargs={
            'drop_name': ['U11', 'ZV100', 'Z25'],
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.001,
            }
        },
        epochs=2,
        batch_size=512,
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 9 / 10,
            'random_state': 43
        },
        scoring=None,
        metrics=['accuracy']
    )
    random = Experiment(
        description='Making an experiemnt to continue for xp_86 for rs_layer10',
        model=mlp_prediction,
        model_kwargs={
            'mlp_kwargs': {
                'sizes': [45],
                'dropout': [0.3],
                'regularizer': [1e-4],
            },
        },
        extractor=PredictionTasks,
        extractor_kwargs={
            'drop_name': ['U11', 'ZV100', 'Z25'],
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.001,
            }
        },
        epochs=2,
        batch_size=512,
        building_model_requires_development_data=True,
        loss='binary_crossentropy',
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 9 / 10,
            'random_state': 43
        },
        scoring=None,
        metrics=['accuracy']
    )

    LR_PREDICTION = Experiment(
        description='Basic logistic regression, to predict if data is before or after vaccination.',
        model=LogisticRegression,
        model_kwargs={
            'class_weight': 'balanced',
            'max_iter': 300
        },
        extractor=PredictionTasks,
        extractor_kwargs={},
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43
        },
        scoring=roc_auc_score,
        metrics=['accuracy']
    )
    xgboost_prediction = Experiment(
        description='Testing to predict the vaccination with XGBOOST',
        model=XGBClassifier,
        model_kwargs={
        },
        extractor=PredictionTasks,
        extractor_kwargs={'drop_name': ['U11', 'ZV100', 'Z25']},
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43
        },
        scoring=None,
        metrics=['accuracy']
    )
    xgboost_prediction_more = Experiment(
        description='Testing to predict the vaccination with XGBOOST',
        model=XGBClassifier,
        model_kwargs={
        },
        extractor=PredictionTasks,
        extractor_kwargs={'drop_name': ['U11', 'ZV100', 'Z25']},
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 10,
            'train_size': 9/10,
            'random_state': 43
        },
        scoring=None,
        metrics=['accuracy']
    )
    random_forest_prediction = Experiment(
        description='Testing to predict the vaccination ',
        model=RandomForestClassifier,
        model_kwargs={
            'n_estimators': 1000,
            'random_state': 44,
            'n_jobs': -1,
            'verbose': 1
        },
        extractor=PredictionTasks,
        extractor_kwargs={'drop_name': ['U11', 'ZV100', 'Z25']},
        cv=GroupShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43
        },
        scoring=None,
        metrics=['accuracy']
    )


class HyperSearch(HyperExperiment, Enum):
    """
    rs_01 = HyperExperiment(
        template=Experiment(
            description='Testing to predict the vaccination ',
            model=mlp_prediction,
            model_kwargs={
                'mlp_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [1500, 1000, 700, 500, 300, 100, 50, 10],
                            k=num_layers,
                            ascending=False
                        ),
                        'dropout': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'regularizer': {
                            "bias": hp.Choices(
                                [1e-2, 1e-3, 1e-4, 0.0],
                                k=num_layers),
                            "activity":hp.Choices(
                                [1e-2, 1e-3, 1e-4, 0.0],
                                k=num_layers),
                            "kernel": hp.Choices(
                                [1e-4, 0.0],
                                k=num_layers)
                        }

                    } for num_layers in [1, 2, 3, 4]
                ]),
            },
            extractor=PredictionTasks,
            extractor_kwargs={
                'drop_name': ['U11', 'ZV100', 'Z25'],
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice(
                                [1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
                }
            },
            epochs=30,
            batch_size=512,
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            cv=GroupShuffleSplit,
            cv_kwargs={
                'n_splits': 1,
                'train_size': 9/10,
                'random_state': 43
            },
            scoring=None,
            metrics=['accuracy', 'auc'],
            save_model=False,
            random_state=hp.Int(0, 1000000000)
        ),
        random_seed=999,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 50
        }
    )
    rs_02 = HyperExperiment(
        template=Experiment(
            description='Testing to predict the vaccination ',
            model=mlp_prediction,
            model_kwargs={
                'mlp_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [1500, 1000, 700, 500, 300, 100, 50, 10],
                            k=num_layers,
                            ascending=False
                        ),
                        'dropout': hp.Choices(
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            k=num_layers
                        ),
                        'regularizer': {
                            "bias": hp.Choices(
                                [1e-2, 1e-3, 1e-4, 0.0],
                                k=num_layers),
                            "activity": hp.Choices(
                                [1e-2, 1e-3, 1e-4, 0.0],
                                k=num_layers),
                            "kernel": hp.Choices(
                                [1e-4, 0.0],
                                k=num_layers)
                        }

                    } for num_layers in [1, 2, 3, 4]
                ]),
            },
            extractor=PredictionTasks,
            extractor_kwargs={
                'drop_name': ['U11', 'ZV100', 'Z25'],
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice(
                        [1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
                }
            },
            epochs=30,
            batch_size=512,
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            cv=GroupShuffleSplit,
            cv_kwargs={
                'n_splits': 1,
                'train_size': 9 / 10,
                'random_state': 43
            },
            scoring=None,
            metrics=['accuracy'],
            save_model=False,
            random_state=hp.Int(0, 1000000000)
        ),
        random_seed=999,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 100
        }
    )
    rs_03 = HyperExperiment(
        template=Experiment(
            description='Testing to predict the vaccination ',
            model=mlp_prediction,
            model_kwargs={
                'mlp_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [10],
                            k=num_layers,
                            ascending=False
                        ),
                        'dropout': hp.Choices(
                            [0.0],
                            k=num_layers
                        ),
                        'regularizer': {
                            "bias": hp.Choices(
                                [1e-2, 1e-3,],
                                k=num_layers),
                            "activity": hp.Choices(
                                [ 1e-4, 0.0],
                                k=num_layers),
                            "kernel": hp.Choices(
                                [1e-4, 0.0],
                                k=num_layers)
                        }

                    } for num_layers in [1, 2]
                ]),
            },
            extractor=PredictionTasks,
            extractor_kwargs={
                'drop_name': ['U11', 'ZV100', 'Z25'],
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice(
                        [1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
                }
            },
            epochs=1,
            batch_size=512,
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            cv=GroupShuffleSplit,
            cv_kwargs={
                'n_splits': 1,
                'train_size': 9 / 10,
                'random_state': 43
            },
            scoring=None,
            metrics=['accuracy', 'auc'],
            save_model=False,
            random_state=hp.Int(0, 1000000000),
            save_train_result=False,
            save_val_result=False,
        ),
        random_seed=99,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 1000
        }
    )
    rs_layer_10 = HyperExperiment(
        template=Experiment(
            description='Testing to predict the vaccination ',
            model=mlp_prediction,
            model_kwargs={
                'mlp_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [5, 10, 20],
                            k=num_layers,
                            ascending=False
                        ),
                        'dropout': hp.Choice(
                            [0.0, 0.1, 0.2, 0.3, 0.4]
                        ),
                        'regularizer': {
                            "activity": hp.Choices(
                                [1e-4, 0.0],
                                k=num_layers),
                            "activity": hp.Choices(
                                [1e-4, 0.0],
                                k=num_layers),
                            "bias": 0.0
                        }

                    } for num_layers in [1, 2, 3]
                ]),
            },
            extractor=PredictionTasks,
            extractor_kwargs={
                'drop_name': ['U11', 'ZV100', 'Z25'],
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice(
                        [3e-3, 1e-3, 3e-4, 1e-4])
                }
            },
            epochs=10,
            batch_size=hp.Choice(
                        [64, 128, 256, 512]),
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            cv=GroupShuffleSplit,
            cv_kwargs={
                'n_splits': 1,
                'train_size': 9 / 10,
                'random_state': 43
            },
            scoring=None,
            metrics=['accuracy', 'auc'],
            save_model=False,
            random_state=hp.Int(0, 1000000000),
            save_train_result=False,
            save_val_result=False
        ),
        random_seed=999,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 100
        }
    )
"""
    rs_layer_100 = HyperExperiment(
        template=Experiment(
            description='Testing to predict the vaccination ',
            model=mlp_prediction,
            model_kwargs={
                'mlp_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [50, 100, 200],
                            k=num_layers,
                            ascending=False
                        ),
                        'dropout': hp.Choice(
                            [0.0, 0.1, 0.2, 0.3, 0.4]
                        ),
                        'regularizer': {
                            "activity": hp.Choices(
                                [1e-4, 0.0],
                                k=num_layers),
                            "kernel": hp.Choices(
                                [1e-4, 0.0],
                                k=num_layers),
                            "bias": 0.0
                        }

                    } for num_layers in [1, 2, 3, 4]
                ]),
            },
            extractor=PredictionTasks,
            extractor_kwargs={
                'drop_name': ['U11', 'ZV100', 'Z25'],
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice(
                        [3e-3, 1e-3, 3e-4, 1e-4])
                }
            },
            epochs=10,
            batch_size=hp.Choice(
                        [64, 128, 256, 512]),
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            cv=GroupShuffleSplit,
            cv_kwargs={
                'n_splits': 1,
                'train_size': 9 / 10,
                'random_state': 43
            },
            scoring=None,
            metrics=['accuracy', 'auc'],
            save_model=False,
            random_state=hp.Int(0, 1000000000),
            save_train_result=False,
            save_val_result=False
        ),
        random_seed=999,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 100
        }
    )
    rs_layer_1000 = HyperExperiment(
        template=Experiment(
            description='Testing to predict the vaccination ',
            model=mlp_prediction,
            model_kwargs={
                'mlp_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [500, 1000, 2000],
                            k=num_layers,
                            ascending=False
                        ),
                        'dropout': hp.Choice(
                            [0.0, 0.1, 0.2, 0.3, 0.4]
                        ),
                        'regularizer': {
                            "activity": hp.Choices(
                                [1e-4, 0.0],
                                k=num_layers),
                            "kernel": hp.Choices(
                                [1e-4, 0.0],
                                k=num_layers),
                            "bias": 0.0
                        }

                    } for num_layers in [1, 2, 3]
                ]),
            },
            extractor=PredictionTasks,
            extractor_kwargs={
                'drop_name': ['U11', 'ZV100', 'Z25'],
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice(
                        [3e-3, 1e-3, 3e-4, 1e-4])
                }
            },
            epochs=10,
            batch_size=hp.Choice(
                [64, 128, 256, 512]),
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            cv=GroupShuffleSplit,
            cv_kwargs={
                'n_splits': 1,
                'train_size': 9 / 10,
                'random_state': 43
            },
            scoring=None,
            metrics=['accuracy', 'auc'],
            save_model=False,
            random_state=hp.Int(0, 1000000000),
            save_train_result=False,
            save_val_result=False
        ),
        random_seed=999,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 100
        }
    )
    rs_layer_3000 = HyperExperiment(
        template=Experiment(
            description='Testing to predict the vaccination ',
            model=mlp_prediction,
            model_kwargs={
                'mlp_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [1500, 3000, 6000],
                            k=num_layers,
                            ascending=False
                        ),
                        'dropout': hp.Choice(
                            [0.0, 0.1, 0.2, 0.3, 0.4]
                        ),
                        'regularizer': {
                            "activity": hp.Choices(
                                [0.0],
                                k=num_layers),
                            "kernel": hp.Choices(
                                [1e-4, 0.0],
                                k=num_layers),
                            "bias": 0.0
                        }

                    } for num_layers in [1, 2, 3]
                ]),
            },
            extractor=PredictionTasks,
            extractor_kwargs={
                'drop_name': ['U11', 'ZV100', 'Z25'],
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice(
                        [3e-3, 1e-3, 3e-4, 1e-4])
                }
            },
            epochs=10,
            batch_size=512,
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            cv=GroupShuffleSplit,
            cv_kwargs={
                'n_splits': 1,
                'train_size': 9 / 10,
                'random_state': 43
            },
            scoring=None,
            metrics=['accuracy', 'auc'],
            save_model=False,
            random_state=hp.Int(0, 1000000000),
            save_train_result=False,
            save_val_result=False
        ),
        random_seed=999,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 100
        }
    )
    rs_layer_10000 = HyperExperiment(
        template=Experiment(
            description='Testing to predict the vaccination ',
            model=mlp_prediction,
            model_kwargs={
                'mlp_kwargs': hp.Choice([
                    {
                        'sizes': hp.SortedChoices(
                            [5000, 10000, 20000],
                            k=num_layers,
                            ascending=False
                        ),
                        'dropout': hp.Choice(
                            [0.0, 0.1, 0.2, 0.3, 0.4]
                        ),
                        'regularizer': {
                            "activity": hp.Choices(
                                [0.0],
                                k=num_layers),
                            "kernel": hp.Choices(
                                [1e-4, 0.0],
                                k=num_layers),
                            "bias": 0.0
                        }

                    } for num_layers in [1, 2, 3]
                ]),
            },
            extractor=PredictionTasks,
            extractor_kwargs={
                'drop_name': ['U11', 'ZV100', 'Z25'],
            },
            optimizer={
                'name': Adam,
                'kwargs': {
                    'learning_rate': hp.Choice(
                        [3e-3, 1e-3, 3e-4, 1e-4])
                }
            },
            epochs=10,
            batch_size=hp.Choice(
                [64, 128, 256, 512]),
            building_model_requires_development_data=True,
            loss='binary_crossentropy',
            cv=GroupShuffleSplit,
            cv_kwargs={
                'n_splits': 1,
                'train_size': 9 / 10,
                'random_state': 43
            },
            scoring=None,
            metrics=['accuracy', 'auc'],
            save_model=False,
            random_state=hp.Int(0, 1000000000),
            save_train_result=False,
            save_val_result=False
        ),
        random_seed=999,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 100
        }
    )
