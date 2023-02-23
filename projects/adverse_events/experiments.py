from enum import Enum

from keras.optimizers import Adam
from sklearn.model_selection import ShuffleSplit

from mim.experiments.experiments import Experiment
from projects.adverse_events.extractor import SelfControl
from projects.adverse_events.models import autoencoder


class AutoEncoder(Experiment, Enum):
    TEST = Experiment(
        description='',
        model=autoencoder,
        model_kwargs={},
        extractor=SelfControl,
        extractor_kwargs={},
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': 0.001,
            }
        },
        epochs=2,
        batch_size=32,
        building_model_requires_development_data=True,
        loss='mse',
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2/3,
            'random_state': 43
        },
        plot_model=False,
        scoring=None
    )
