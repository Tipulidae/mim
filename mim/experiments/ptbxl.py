from enum import Enum

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ShuffleSplit

from mim.experiments.experiments import Experiment
from mim.extractors.ptbxl import PTBXL
from mim.models.simple_nn import ptbxl_cnn


class ptbxl(Experiment, Enum):
    TEST = Experiment(
        description="",
        model=ptbxl_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'down_sample': True,
                'num_layers': 2,
                'dropout': 0.3,
            },
            'ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.3],
                'batch_norms': [False],
                'activity_regularizer': 0.01,
                'kernel_regularizer': 0.01,
                'bias_regularizer': 0.01,
            },
        },
        extractor=PTBXL,
        extractor_kwargs={
            'index': {'size': 'XL'}
        },
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2/3,
            'random_state': 43
        },
        scoring=roc_auc_score,
        metrics=['accuracy', 'auc']
    )
