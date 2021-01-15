from enum import Enum

from sklearn.metrics import roc_auc_score

from mim.experiments.experiments import Experiment
from mim.model_wrapper import KerasWrapper
from mim.extractors.esc_trop import EscTrop
from mim.models.simple_nn import BasicCNN
from mim.cross_validation import ChronologicalSplit


# Here's an attempt at a structure for experiment names:
# [Data source]_[[features]]_[target]_[model]_[version]
# Data source: {ESC, EXPECT, PTB, PTBXL, ...}
# Features: {B, R}#, where B=beat, R=raw and # is the number of records used.
# Other common feature-sets to be given names as needed.
# Target: {MACE,AMI,...}#, where # would indicate some time frame
# Model: Some short hand for the models that I use
# Version: If I run multiple variations of this experiment, a version is handy


class MultipleECG(Experiment, Enum):
    ESC_B1_MACE30_BCNN2_V1 = Experiment(
        description='Baseline CNN model using only current ECG median beat to '
                    'predict MACE within 30 days.',
        algorithm=KerasWrapper,
        params={
            'model': BasicCNN,
            'num_conv_layers': 2,
            'input_shape': (1200, 8),
            'epochs': 10,
            'batch_size': 64
        },
        extractor=EscTrop,
        features={'ecg_mode': 'beat'},
        index={},
        cv=ChronologicalSplit,
        cv_args={
            'test_size': 0.667
        },
        hold_out_size=0.25,
        scoring=roc_auc_score,
    )

    ESC_R1_MACE30_BCNN2_V1 = Experiment(
        description='Baseline CNN model using only current raw ECG signal to '
                    'predict MACE within 30 days.',
        algorithm=KerasWrapper,
        params={
            'model': BasicCNN,
            'num_conv_layers': 2,
            'input_shape': (10000, 8),
            'epochs': 200,
            'batch_size': 64
        },
        extractor=EscTrop,
        features={'ecg_mode': 'raw'},
        index={},
        cv=ChronologicalSplit,
        cv_args={
            'test_size': 0.667
        },
        hold_out_size=0.25,
        scoring=roc_auc_score,
    )

    ESC_R1_MACE30_BCNN3_V1 = ESC_R1_MACE30_BCNN2_V1._replace(
        description='Uses 3 conv layers instead of just 2.',
        params={
            'model': BasicCNN,
            'num_conv_layers': 3,
            'input_shape': (10000, 8),
            'epochs': 200,
            'batch_size': 64
        },
    )
