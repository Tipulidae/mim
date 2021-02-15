# -*- coding: utf-8 -*-

from enum import Enum
from sklearn.metrics import roc_auc_score
from os.path import join
import tensorflow as tf

from mim.experiments.experiments import Experiment
from mim.config import GLUCOSE_ROOT
from mim.extractors.ab_json import ABJSONExtractor
import mim.models.ab_nn as ab_nn


class ABGlucose(Experiment, Enum):

    _BASE = Experiment(
        description="Abstract. Don't run me.",
        scoring=roc_auc_score,
        extractor=ABJSONExtractor,
        extractor_kwargs={
            "index": {"train": join(GLUCOSE_ROOT, "hbg+lund-train.json.gz"),
                      "val": join(GLUCOSE_ROOT, "hbg+lund-dev.json.gz"),
                      "test": join(GLUCOSE_ROOT, "hbg+lund-test.json.gz")},
            "features": {"gender",
                         "age",
                         "bl-Glukos", "bl-TnT", "bl-Krea", "bl-Hb"},
            "labels": {"target": "label-index+30d-ami+death-30d"},
        }
    )

    KERAS_LR_BLOOD_BL = _BASE._replace(
        description="Log Reg baseline with Keras, using age, "
                    "gender, + 4 blood samples",
        model=ab_nn.ab_simple_lr,
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

    TEMP = KERAS_LR_BLOOD_BL._replace(
        description="Foo",
        ignore_callbacks=False,
        epochs=10
    )
