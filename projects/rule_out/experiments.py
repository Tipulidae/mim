from enum import Enum

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from keras.optimizers import SGD
from keras.losses import BinaryCrossentropy, Hinge

from mim.experiments.experiments import Experiment
from mim.losses.dynamic_weights import (
    EdenLoss,
    EdenLossV2,
    EdenLossV3
)
from projects.rule_out.extractor import Blobs, Cones
from projects.rule_out.models import single_layer_perceptron, \
    multi_layer_perceptron


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
    LR_B2D_V1_BCE = Experiment(
        description='Eban dataset',
        model=LogisticRegression,
        model_kwargs={
            'max_iter': 200,
            'penalty': None,
            'C': 1.0
        },
        extractor=Blobs,
        extractor_kwargs={
            'features': {
                'negatives_counts': [1500, 750],
                'positives_counts': [1800, 200],
                'negatives_centers': [(0.0, -0.5), (1.0, 1.0)],
                'positives_centers': [(0.0, 1.0), (1.0, -0.5)],
                'negatives_std': [0.15, 0.1],
                'positives_std': [0.15, 0.1]
            },
        },
        # reduce_lr_on_plateau={
        #     'factor': 0.1,
        #     'patience': 10,
        # },
        cv=StratifiedShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 0.5,
            'random_state': 43,
        },
        scoring=roc_auc_score,
    )
    SLP_B2D_V1_BCE = Experiment(
        description='Eban dataset',
        model=single_layer_perceptron,
        model_kwargs={},
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1.0}
        },
        loss=Hinge,
        loss_kwargs={},
        epochs=200,
        batch_size=-1,
        building_model_requires_development_data=True,
        extractor=Blobs,
        extractor_kwargs={
            'features': {
                'negatives_counts': [1500, 500],
                'positives_counts': [1800, 200],
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
    SLP_B2D_V1_EDEN = SLP_B2D_V1_BCE._replace(
        description='Eden loss',
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1.0}
        },
        epochs=1000,

        loss=EdenLoss,
        loss_kwargs={
            'target_tpr': 0.95,
            'tpr_weight': 2.0
        },
        verbose=0
    )
    SLP_B2D_V1_EDEN2 = SLP_B2D_V1_BCE._replace(
        description='Eden loss, v2',
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 0.1}
        },
        epochs=1000,
        loss=EdenLossV2,
        loss_kwargs={
            'target_tpr': 0.95,
            'tpr_weight': 20.0
        },
        verbose=0
    )
    SLP_B2D_V1_EDEN3 = SLP_B2D_V1_BCE._replace(
        description='Eden loss, v3',
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1.0}
        },
        epochs=600,
        loss=EdenLossV3,
        loss_kwargs={
            'target_tpr': 0.95,
            'alpha': 5.0,
            'beta': 2.0
        },
        verbose=0
    )
    SLP_B2D_V2_EDEN3 = SLP_B2D_V1_BCE._replace(
        description='Eden loss, v3, mickey-mouse data',
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1.0}
        },
        extractor_kwargs={
            'features': {
                'positives_counts': [500, 500, 9000],
                'negatives_counts': [10000],
                'positives_centers': [(-1.0, 2.0), (1.0, 2.0), (0.0, -2.0)],
                'negatives_centers': [(0.0, 0.0)],
                'positives_std': [0.1, 0.1, 0.3],
                'negatives_std': [0.5],
            },
        },
        epochs=600,
        loss=EdenLossV3,
        loss_kwargs={
            'target_tpr': 0.95,
            'alpha': 5.0,
            'beta': 2.0
        },
        verbose=0
    )
    SLP_B2D_V2_BCE = Experiment(
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
        class_weight={1: 5.0, 0: 1.0},
        building_model_requires_development_data=True,
        extractor=Blobs,
        extractor_kwargs={
            'features': {
                'positives_counts': [500, 500, 9000],
                'negatives_counts': [10000],
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
    EDEN_B2D = Experiment(
        description='Trying to get EdenLoss to behave',
        model=single_layer_perceptron,
        model_kwargs={},
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1}
        },
        loss=EdenLoss,
        loss_kwargs={'target_tpr': 0.95},
        epochs=200,
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
    EDEN_CONES = Experiment(
        description='Trying to get EdenLoss to behave',
        model=multi_layer_perceptron,
        model_kwargs={},
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 0.1}
        },
        loss=EdenLoss,
        loss_kwargs={'target_tpr': 0.95},
        epochs=200,
        batch_size=1024,
        building_model_requires_development_data=True,
        extractor=Cones,
        extractor_kwargs={
            'features': {
                'positives_count': 20000,
                'negatives_count': 2000,
                'positives_center': (0.0, -1.0),
                'negatives_center': (0.0, 1.0),
                'positives_scale': (2.0, 2.0),
                'negatives_scale': (2.0, 2.0),
                'cheat': True
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
    MLP_CONES = Experiment(
        description='For comparison with Eden',
        model=multi_layer_perceptron,
        model_kwargs={},
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 0.01}
        },
        loss='binary_crossentropy',
        # loss_kwargs={'target_tpr': 0.95},
        epochs=100,
        batch_size=1024,
        building_model_requires_development_data=True,
        extractor=Cones,
        extractor_kwargs={
            'features': {
                'positives_count': 20000,
                'negatives_count': 2000,
                'positives_center': (0.0, -1.0),
                'negatives_center': (0.0, 1.0),
                'positives_scale': (2.0, 2.0),
                'negatives_scale': (2.0, 2.0),
                'cheat': True
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
