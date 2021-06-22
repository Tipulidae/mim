from enum import Enum

from sklearn.metrics import roc_auc_score

from mim.experiments.experiments import Experiment
from mim.model_wrapper import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from mim.extractors.expect import Expect
from mim.extractors.ptbxl import PTBXL
from mim.models.simple_nn import ecg_cnn


class MyocardialInfarction(Experiment, Enum):
    THAN_EXPECT_GB = Experiment(
        description='Gradient Boosting Classifier similar to that of '
                    'Than et. al, Circulation 2019',
        model=GradientBoostingClassifier,
        scoring=roc_auc_score,

        model_kwargs={
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 2,
            'min_samples_leaf': 7,
            'subsample': 0.5
        },
        extractor=Expect,
        extractor_kwargs={
            "index": {'source': 'two_tnt_gen-all.json'},
            "labels": {'target': 'label-index-mi'},
            "features": {'troponin',
                         'age',
                         'gender'},
        },
    )

    THAN_EXPECT_RF = THAN_EXPECT_GB._replace(
        description='Replaces gradient boosting with random forest',
        model=RandomForestClassifier,
        model_kwargs={'n_estimators': 1000}
    )

    THAN_EXPECT_RF2 = THAN_EXPECT_RF._replace(
        description='Classifies index MI using additional features',
        extractor_kwargs=THAN_EXPECT_RF.extractor_kwargs.copy().update(
            {"features": {
                'troponin',
                'previous_conditions',
                'age',
                'gender'
            }}
        )
    )

    PTBXL_SMALL = Experiment(
        description='Small experiment using a CNN with PTBXL',
        model=ecg_cnn,
        model_kwargs={
            'num_conv_layers': 2,
            'input_shape': (1000, 12),
            'epochs': 5,
            'batch_size': 32
        },
        extractor=PTBXL,
        extractor_kwargs={
            "index": {'size': 'XSS'}
        },
        scoring=roc_auc_score,
    )
