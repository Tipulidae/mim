from enum import Enum

import torch
from keras.optimizers import Adam
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score

from mim.experiments.experiments import Experiment
from mim.fakes.fake_extractors import FakeExtractor
from mim.models.load import load_torch_model
from mim.models.util import cosine_decay_with_warmup_torch
from projects.transfer.models import simple_mlp_tf, simple_mlp_pt


class PytorchTensorflowSanity(Experiment, Enum):
    SIMPLE_TF = Experiment(
        description='',
        extractor=FakeExtractor,
        extractor_index={
            'n_samples': 2048*32,
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
        batch_size=64,
        loss='binary_crossentropy',
        scoring=roc_auc_score,
        metrics=['auc'],
        cv=ShuffleSplit,
        cv_kwargs={
            'n_splits': 1,
            'train_size': 0.5,
            'random_state': 222
        },
        use_tensorboard=True,
        save_val_pred_history=True,
        save_train_pred_history=True
    )
    SIMPLE_PT = Experiment(
        description='',
        extractor=FakeExtractor,
        extractor_index={
            'n_samples': 2048*32,
            'n_features': 100,
            'n_informative': 20,
            'n_redundant': 50,
            'class_sep': 0.1,
            'n_classes': 2,
            'n_clusters_per_class': 5,
            'flip_y': 0.1,
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
        batch_size=64,
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
        save_model=True,
        model_checkpoints=True,  # For now, saves model after each epoch
        save_learning_rate=False,
        save_val_pred_history=True,
        save_train_pred_history=True,
    )
    PRETRAINED_PT = Experiment(
        description='',
        extractor=FakeExtractor,
        extractor_index={
            'n_samples': 2048*32,
            'n_features': 100,
            'n_informative': 20,
            'n_redundant': 50,
            'class_sep': 0.1,
            'n_classes': 2,
            'n_clusters_per_class': 5,
            'flip_y': 0.1,
            'random_state': 123
        },
        model=load_torch_model,
        model_kwargs={
            'base_path': 'data/test_results/transfer/'
                         'PytorchTensorflowSanity/SIMPLE_PT',
            'split_number': 0,
        },
        data_fits_in_memory=True,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={
            'weight_decay': 0.01
        },
        learning_rate=1e-3,
        epochs=10,
        batch_size=64,
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
        save_model=True,
        save_learning_rate=False,
        save_val_pred_history=True,
        save_train_pred_history=True,
    )

    PT_LR_SCHEDULE = Experiment(
        description='',
        extractor=FakeExtractor,
        extractor_index={
            'n_samples': 2048*8,
            'n_features': 100,
            'n_informative': 20,
            'n_redundant': 50,
            'class_sep': 0.1,
            'n_classes': 2,
            'n_clusters_per_class': 5,
            'flip_y': 0.1,
            'random_state': 123
        },
        model=simple_mlp_pt,
        model_kwargs={},
        data_fits_in_memory=True,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={
            'weight_decay': 0.01,
        },
        learning_rate={
            'scheduler': cosine_decay_with_warmup_torch,
            'kwargs': {
                'initial_learning_rate': 0.2,
                'warmup_target': 1,
                'alpha': 0.01,
                'warmup_epochs': 5,
                'decay_epochs': 5
            }
        },
        epochs=15,
        batch_size=64,
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
        save_model=True,
        save_learning_rate=True,
        save_val_pred_history=True,
        save_train_pred_history=True,
    )
