# -*- coding: utf-8 -*-

from enum import Enum
from os.path import join
import tensorflow as tf

from mim.experiments.experiments import Experiment
from mim.config import GLUCOSE_ROOT
from mim.extractors.ab_json import ABJSONExtractor
import mim.models.ab_nn as ab_nn

BASE_EXTRACTOR_KWARGS: dict = {
        "index": {"train": join(GLUCOSE_ROOT, "hbg+lund-train.json.gz"),
                  "val": join(GLUCOSE_ROOT, "hbg+lund-dev.json.gz"),
                  "test": join(GLUCOSE_ROOT, "hbg+lund-test.json.gz")},
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
            "features": {"gender",
                         "age",
                         "bl-Glukos", "bl-TnT", "bl-Krea", "bl-Hb"}
        }),
        building_model_requires_development_data=True,
        ignore_callbacks=False,
        model_kwargs={},
        epochs=300,
        batch_size=-1,
        optimizer={
            'name': tf.keras.optimizers.SGD,
            'kwargs': {'learning_rate': 1}
        },
        data_provider_kwargs={
            "mode": "train_val"
        }
    )

    KERAS_HIDDEN_10 = Experiment(
        description="Neural network with one hidden layer, 10 neurons"
                    "simple inputs",
        extractor=ABJSONExtractor,
        extractor_kwargs=copy_update_dict(BASE_EXTRACTOR_KWARGS, {
            "features": {"gender",
                         "age",
                         "bl-Glukos", "bl-Krea", "bl-Hb"}
        }),
        building_model_requires_development_data=True,
        ignore_callbacks=False,
        model=ab_nn.ab_simple_one_hidden_layer,
        model_kwargs={
            "hidden_layer_n": 10,
        },
        epochs=200,
        batch_size=32,
        optimizer={
            'name': tf.keras.optimizers.Adam,
            'kwargs': {'learning_rate': 0.003}
        },
        data_provider_kwargs={
            "mode": "train_val"
        }
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
        },
        data_provider_kwargs={
            "mode": "train_val"
        }
    )
