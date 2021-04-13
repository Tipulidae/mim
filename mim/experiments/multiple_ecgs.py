from enum import Enum

from sklearn.metrics import roc_auc_score

from mim.experiments.experiments import Experiment
from mim.extractors.esc_trop import EscTrop
from mim.models.simple_nn import basic_cnn
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
        model=basic_cnn,
        model_kwargs={
            'num_conv_layers': 2,
        },
        epochs=200,
        batch_size=64,
        optimizer='sgd',
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'beat',
                'ecgs': ['index']
            },
            "index": {}
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1/3},
        scoring=roc_auc_score,
    )

    FOO = ESC_B1_MACE30_BCNN2_V1._replace(
        description='Foo'
    )

    # AB: Shouldn't this jsut be a copy of the above experiment, with the
    # features changed?
    # SANITY1 = Experiment(
    #     description='Try to predict mace 30 using only the old ecg...',
    #     model=basic_cnn,
    #     model_kwargs={
    #         'num_conv_layers': 2,
    #         'epochs': 200,
    #         'batch_size': 64
    #     },
    #     extractor=EscTrop,
    #     features={
    #         'ecg_mode': 'beat',
    #         'ecgs': ['old']
    #     },
    #     scoring=roc_auc_score,
    # )

    ESC_B1_MACE30_BCNN2_V2 = ESC_B1_MACE30_BCNN2_V1._replace(
        model_kwargs={
            'num_conv_layers': 2,
            'epochs': 200,
            'batch_size': 64,
        },
    )

    ESC_B1AS_MACE30_BCNN2_V1 = ESC_B1_MACE30_BCNN2_V1._replace(
        description='Baseline CNN model using a 2 conv layer network on '
                    '1 ECG median beat plus age and sex features concatenated '
                    'at the end, predicting MACE within 30 days.',
        model_kwargs={
            'num_conv_layers': 2,
            'epochs': 200,
            'batch_size': 64
        },
        extractor_kwargs=ESC_B1_MACE30_BCNN2_V1.extractor_kwargs.copy().update(
            {"features": {
                'ecg_mode': 'beat',
                'ecgs': ['index'],
                'features': ['age', 'sex']
            }}
        )
    )

    ESC_B2AS_MACE30_BCNN2_V1 = ESC_B1_MACE30_BCNN2_V1._replace(
        description='Running two CNNs in parallel on two ECG beat signals. '
                    'Also uses age and sex as features.',
        model_kwargs={
            'num_conv_layers': 2,
            'epochs': 200,
            'batch_size': 64
        },
        extractor_kwargs=ESC_B1_MACE30_BCNN2_V1.extractor_kwargs.copy().update(
            {"features": {
                'ecg_mode': 'beat',
                'ecgs': ['index', 'old'],
                'features': ['age', 'sex']
            }}
        )
    )
