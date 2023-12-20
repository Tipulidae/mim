from enum import Enum

import torch
from keras.optimizers import Adam
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score

from mim.experiments.experiments import Experiment
from mim.fakes.fake_extractors import FakeExtractor
from projects.transfer.models import simple_mlp_tf, simple_mlp_pt


class PytorchTensorflowSanity(Experiment, Enum):
    SIMPLE_TF = Experiment(
        description='',
        extractor=FakeExtractor,
        extractor_index={
            'n_samples': 2048,
            'n_features': 100,
            'n_informative': 20,
            'n_redundant': 50,
            'class_sep': 2.0,
            'n_classes': 2,
            'n_clusters_per_class': 5,
            'random_state': 123
        },
        model=simple_mlp_tf,
        model_kwargs={},
        data_fits_in_memory=True,
        optimizer=Adam,
        learning_rate=1e-3,
        epochs=10,
        batch_size=16,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        metrics=['auc'],
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 0.5,
            'random_state': 222
        },
        use_tensorboard=False,
        save_learning_rate=False,
        save_val_pred_history=False
    )
    SIMPLE_PT = Experiment(
        description='',
        extractor=FakeExtractor,
        extractor_index={
            'n_samples': 2048,
            'n_features': 100,
            'n_informative': 20,
            'n_redundant': 50,
            'class_sep': 2.0,
            'n_classes': 2,
            'n_clusters_per_class': 5,
            'random_state': 123
        },
        model=simple_mlp_pt,
        model_kwargs={},
        data_fits_in_memory=True,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={
            'weight_decay': 0.01
        },
        learning_rate=1e-3,
        epochs=10,
        batch_size=16,
        loss=torch.nn.BCELoss,
        loss_kwargs={},
        scoring=roc_auc_score,
        metrics=['auc'],
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 0.5,
            'random_state': 222
        },
        use_tensorboard=False,
        save_learning_rate=False,
        save_val_pred_history=False
    )
