from enum import Enum

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from keras.optimizers import SGD
from keras.losses import BinaryCrossentropy

from mim.experiments.experiments import Experiment
from projects.rule_out.extractor import Blobs
from projects.rule_out.models import single_layer_perceptron


class RuleOut(Experiment, Enum):
    LR_B1D = Experiment(
        description='',
        model=LogisticRegression,
        model_kwargs={},
        extractor=Blobs,
        # extractor_kwargs={
        #     'index': {'n_samples': 2000},
        #     'features': {
        #         'n_features': 1,
        #         'centers': [[-1.0], [1.0]],
        #         'cluster_std': [1.0, 1.0]
        #     },
        # },
        extractor_kwargs={
            'features': {
                'negatives_counts': [1000],
                'positives_counts': [1000],
                'negatives_centers': [[-1.0]],
                'positives_centers': [[1.0]],
                'negatives_std': [1.0],
                'positives_std': [1.0]
            },
        },
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43,
        },
        scoring=roc_auc_score,
    )
    LR_B1D_V2 = Experiment(
        description='',
        model=LogisticRegression,
        model_kwargs={},
        extractor=Blobs,
        extractor_kwargs={
            'features': {
                'negatives_counts': [36000],
                'positives_counts': [4000],
                'negatives_centers': [0.0],
                'positives_centers': [2.0],
                'negatives_std': [2.0],
                'positives_std': [1.5]
            },
        },
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 2 / 3,
            'random_state': 43,
        },
        scoring=roc_auc_score,
    )
    LR_B2D_V1 = Experiment(
        description='Same dataset as used by Eban et al in the global-'
                    'objectives paper.',
        model=LogisticRegression,
        model_kwargs={},
        extractor=Blobs,
        extractor_kwargs={
            'features': {
                'negatives_counts': [6000, 200],
                'positives_counts': [1000, 100],
                'negatives_centers': [(0.0, -0.5), (1.0, 1.0)],
                'positives_centers': [(0.0, 1.0), (1.0, -0.5)],
                'negatives_std': [0.15, 0.1],
                'positives_std': [0.15, 0.1]
            },
        },
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 0.5,
            'random_state': 43,
        },
        scoring=roc_auc_score,
    )
    LR_B2D_V2 = Experiment(
        description='1 large negative blob in the middle, 1 large positive '
                    'blob underneath, and two small satellite blobs above. '
                    'Dataset should be such that 95% recall is only possible '
                    'by including the large positive blob plus one of the '
                    'satellites. With a linear classifier, this is impossible '
                    'without getting a large number of false positives.',
        model=LogisticRegression,
        model_kwargs={},
        extractor=Blobs,
        extractor_kwargs={
            'features': {
                'positives_counts': [100, 100, 1800],
                'negatives_counts': [2000],
                'positives_centers': [(-1.0, 2.0), (1.0, 2.0), (0.0, -2.0)],
                'negatives_centers': [(0.0, 0.0)],
                'positives_std': [0.1, 0.1, 0.3],
                'negatives_std': [0.5],
            },
        },
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 0.5,
            'random_state': 43,
        },
        scoring=roc_auc_score,
    )
    SLP_B2D_V2 = Experiment(
        description='The 2D 4-blobs data, but with a single-layer-perceptron '
                    'instead of LR.',
        model=single_layer_perceptron,
        model_kwargs={},
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1.0}
        },
        loss=BinaryCrossentropy,
        loss_kwargs={},
        epochs=100,
        batch_size=-1,
        building_model_requires_development_data=True,
        extractor=Blobs,
        extractor_kwargs={
            'features': {
                'positives_counts': [100, 100, 1800],
                'negatives_counts': [2000],
                'positives_centers': [(-1.0, 2.0), (1.0, 2.0), (0.0, -2.0)],
                'negatives_centers': [(0.0, 0.0)],
                'positives_std': [0.1, 0.1, 0.3],
                'negatives_std': [0.5],
            },
        },
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 0.5,
            'random_state': 43,
        },
        scoring=roc_auc_score,
    )
    TEST = Experiment(
        description='Testing if I can get the custom loss function to work '
                    'at all.',
        model=single_layer_perceptron,
        model_kwargs={},
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 0.1}
        },
        # loss=CustomBCE,

        loss_kwargs={'regularization_factor': 2.0},
        epochs=300,
        batch_size=-1,
        building_model_requires_development_data=True,
        extractor=Blobs,
        extractor_kwargs={
            'features': {
                'positives_counts': [100, 100, 1800],
                'negatives_counts': [2000],
                'positives_centers': [(-1.0, 2.0), (1.0, 2.0), (0.0, -2.0)],
                'negatives_centers': [(0.0, 0.0)],
                'positives_std': [0.1, 0.1, 0.3],
                'negatives_std': [0.5],
            },
        },
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 0.5,
            'random_state': 43,
        },
        scoring=roc_auc_score,
    )
