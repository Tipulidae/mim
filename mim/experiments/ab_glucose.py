# -*- coding: utf-8 -*-

from enum import Enum
from sklearn.metrics import roc_auc_score


from mim.cross_validation import ChronologicalSplit
from mim.experiments.experiments import Experiment
from mim.extractors.ab_json import ABJSONExtractor
import mim.models.ab_nn as ab_nn


class ABGlucose(Experiment, Enum):
    KERAS_LR_BLOOD_BL = Experiment(
        description="Log Reg baseline with Keras, using age, "
                    "gender, + 4 blood samples",
        model=ab_nn.ab_simple_lr,
        model_kwargs={
            'epochs': 100,
            'batch_size': 32,
            # 'learning_rate': 0.01,
            'compile_args': {'optimizer': 'sgd'}
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
        cv_kwargs={"test_size": 0.25},
        scoring=roc_auc_score
    )
