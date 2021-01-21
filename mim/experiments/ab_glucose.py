# -*- coding: utf-8 -*-

from enum import Enum

from mim.experiments.experiments import Experiment
from mim.model_wrapper import KerasWrapper
from mim.extractors.ab_json import ABJSONExtractor


class ABGlucose(Experiment, Enum):
    KERAS_LR_BLOOD_BL = Experiment(
        description="Log Reg baseline with Keras, using age, "
                    "gender, + 4 blood samples",
        algorithm=KerasWrapper,
        params={
            'model': None,
            'epochs': 100,
            'batch_size': -1,
            'optimizer': 'sgd',
            'learning_rate': 0.01
        },
        extractor=ABJSONExtractor,
        index={"json_train": "/home/anders/PycharmProjects/Expect/json_data" +
                             "/pontus_glukos/hbg+lund-train.json.gz"
               },
        features={
            "gender",
            "age",
            "bl-Glukos", "bl-TnT", "bl-Krea", "bl-Hb"
        },
        labels={"target": "label-index+30d-ami+death-30d"}
    )
