from enum import Enum

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from mim.experiments.experiments import Experiment
from mim.model_wrapper import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    KerasWrapper,
)
from mim.extractors.expect import Expect
from mim.extractors.ptbxl import PTBXL
from mim.models.simple_nn import BasicCNN


class MyocardialInfarction(Experiment, Enum):
    THAN_EXPECT_GB = Experiment(
        description='Gradient Boosting Classifier similar to that of '
                    'Than et. al, Circulation 2019',
        algorithm=GradientBoostingClassifier,
        params={
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 2,
            'min_samples_leaf': 7,
            'subsample': 0.5
        },
        features={
            'troponin',
            'age',
            'gender'
        },
        labels={'target': 'label-index-mi'},
        index={'source': 'two_tnt_gen-all.json'},
        cv=KFold,
        cv_args={'n_splits': 5},
        scoring=roc_auc_score,
        extractor=Expect
    )

    THAN_EXPECT_RF = THAN_EXPECT_GB._replace(
        description='Replaces gradient boosting with random forest',
        algorithm=RandomForestClassifier,
        params={'n_estimators': 1000}
    )

    THAN_EXPECT_RF2 = THAN_EXPECT_RF._replace(
        description='Classifies index MI using additional features',
        features={
            'troponin',
            'previous_conditions',
            'age',
            'gender'
        },
    )

    PTBXL_SMALL = Experiment(
        description='Small experiment using a CNN with PTBXL',
        algorithm=KerasWrapper,
        params={'model': BasicCNN},
        extractor=PTBXL,
        index={'size': 'XS'},
        cv=KFold,
        cv_args={'n_splits': 2},
        scoring=roc_auc_score,
    )
