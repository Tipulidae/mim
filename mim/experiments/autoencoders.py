# -*- coding: utf-8 -*-

from enum import Enum

from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

from mim.experiments.experiments import Experiment
from mim.extractors.esc_trop import EscTropECG
from mim.models.ab_nn import dyn_cnn
from mim.cross_validation import ChronologicalSplit


class GenderPredict(Experiment, Enum):
    FOO = Experiment(
        description="Predict gender from ECGs, take one",
        model=dyn_cnn,
        model_kwargs={
            "conv_dropout": [0.0, 0.3, 0.0],
            "conv_filters": [64, 16, 8],
            "conv_kernel_size": [64, 16, 16],
            "conv_pool_size": [32, 4, 8],
            "conv_weight_decay": [0.0001, 0.001, 0.0001],
            "conv_final_dense_neurons": 10,
            "final_dense_neurons": 10,
            "final_dense_dropout": 0.5,
            "activation": "relu",
            "skip_basic": True,
            "ecg_normalization_layer": "PredefinedLambda",
        },

        batch_size=128,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 0.0003}
        },
        epochs=200,

        extractor=EscTropECG,
        extractor_kwargs={"fits_in_memory": False},
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
