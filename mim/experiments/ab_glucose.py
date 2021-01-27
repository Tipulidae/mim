# -*- coding: utf-8 -*-

from enum import Enum
from sklearn.metrics import roc_auc_score
import tensorflow as tf

from mim.cross_validation import ChronologicalSplit
from mim.experiments.experiments import Experiment
from mim.extractors.ab_json import ABJSONExtractor
import mim.models.ab_nn as ab_nn


class ABGlucose(Experiment, Enum):
    KERAS_LR_BLOOD_BL = Experiment(
        description="Log Reg baseline with Keras, using age, "
                    "gender, + 4 blood samples",
        model=ab_nn.ab_simple_lr,
        building_model_requires_development_data=True,
        ignore_callbacks=True,
        model_kwargs={},
        epochs=10,
        batch_size=32,
        optimizer={
            'name': tf.keras.optimizers.SGD,
            'kwargs': {}
        },
        extractor=ABJSONExtractor,
        index={"json_train": "/home/sapfo/andersb/PycharmProjects/Expect/"
                             "json_data/pontus_glukos/hbg+lund-train.json.gz"
               },
        features={
            "gender",
            "age",
            "bl-Glukos", "bl-TnT", "bl-Krea", "bl-Hb"
        },
        labels={"target": "label-index+30d-ami+death-30d"},
        cv=ChronologicalSplit,
        cv_kwargs={"test_size": 0.33},
        scoring=roc_auc_score
    )
