# -*- coding: utf-8 -*-

from enum import Enum
from os.path import join

import tensorflow as tf
from sklearn.model_selection import KFold

from mim.cross_validation import PredefinedSplitsRepeated
from mim.experiments.experiments import Experiment
from mim.config import GLUCOSE_ROOT
from mim.extractors.ab_json import ABJSONExtractor
import mim.models.ab_nn as ab_nn

TRAIN_FILE = join(GLUCOSE_ROOT, "hbg+lund-train.json.gz")
VAL_FILE = join(GLUCOSE_ROOT, "hbg+lund-dev.json.gz")
TEST_FILE = join(GLUCOSE_ROOT, "hbg+lund-test.json.gz")

BASE_EXTRACTOR_KWARGS: dict = {
    "index": {
        "train_files": [TRAIN_FILE],
        "val_files": [VAL_FILE],
    },
    "labels": {"target": "label-index+30d-ami+death-30d"},
}


def copy_update_dict(base_dict: dict, updates: dict) -> dict:
    r = base_dict.copy()
    r.update(updates)
    return r


class ABGlucose(Experiment, Enum):
    KERAS_LR_BLOOD_BL = Experiment(
        description="Log Reg baseline with Keras, using age, "
                    "gender, + 4 blood samples",
        model=ab_nn.ab_simple_lr,
        extractor=ABJSONExtractor,
        extractor_kwargs=copy_update_dict(BASE_EXTRACTOR_KWARGS, {
            "features": {"gender", "age", "bl-Glukos", "bl-TnT", "bl-Krea",
                         "bl-Hb"}
        }),
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        ignore_callbacks=False,
        model_kwargs={},
        epochs=300,
        batch_size=-1,
        optimizer={
            'name': tf.keras.optimizers.SGD,
            'kwargs': {'learning_rate': 1}
        },
    )

    EXAMPLE_USING_KFOLD_ON_VAL_TEST = KERAS_LR_BLOOD_BL._replace(
        description="Small example showing how to use a different data set.",
        extractor_kwargs=copy_update_dict(
            BASE_EXTRACTOR_KWARGS,
            {
                "features": {
                    "gender",
                    "age",
                    "bl-Glukos",
                    "bl-TnT",
                    "bl-Krea",
                    "bl-Hb"
                },
                "index": {
                    "train_files": [VAL_FILE, TEST_FILE]
                }
            }
        ),
        cv=KFold,
        cv_kwargs={'num_folds': 5},
        use_predefined_splits=False
    )

    KERAS_HIDDEN_10 = Experiment(
        description="Neural network with one hidden layer, 10 neurons"
                    "simple inputs",
        extractor=ABJSONExtractor,
        extractor_kwargs=copy_update_dict(BASE_EXTRACTOR_KWARGS, {
            "features": {"gender",
                         "age",
                         "bl-Glukos", "bl-Krea", "bl-Hb", "bl-TnT"}
        }),
        building_model_requires_development_data=True,
        ignore_callbacks=True,
        model=ab_nn.ab_simple_one_hidden_layer,
        model_kwargs={
            "hidden_layer_n": 10,
            "l2": 0.01,
            "dense_dropout": 0.4
        },
        epochs=300,
        batch_size=32,
        optimizer={
            'name': tf.keras.optimizers.Adam,
            'kwargs': {'learning_rate': 0.001}
        },
        use_predefined_splits=True,
        cv=PredefinedSplitsRepeated,
        cv_kwargs={'repeats': 5}
    )

    ECG_BEAT = Experiment(
        description="ECG beat model",
        model=ab_nn.dyn_cnn,
        extractor=ABJSONExtractor,
        extractor_kwargs=copy_update_dict(BASE_EXTRACTOR_KWARGS, {
            "features": {"gender",
                         "age",
                         "bl-Glukos", "bl-TnT", "bl-Krea", "bl-Hb",
                         "ecg_beat_8"}
        }),
        use_predefined_splits=True,
        building_model_requires_development_data=True,
        ignore_callbacks=False,
        model_kwargs={
            "conv_dropout": [0.1, 0.5],
            "conv_filters": [8, 16],
            "conv_kernel_size": [8, 32],
            "conv_pool_size": [16, 16],
            "conv_weight_decay": [0.0, 0.01],
            "conv_final_dense_neurons": 10,
            "final_dense_neurons": 20,
            "final_dense_dropout": 0.3,
            "activation": "relu",
        },
        epochs=300,
        batch_size=32,
        optimizer={
            'name': tf.keras.optimizers.Adam,
            'kwargs': {'learning_rate': 0.003}
        }
    )
