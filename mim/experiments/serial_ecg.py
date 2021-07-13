from enum import Enum

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from sklearn.metrics import roc_auc_score

from mim.experiments.experiments import Experiment
from mim.extractors.esc_trop import EscTrop
from mim.models.simple_nn import ecg_cnn, ffnn
from mim.models.load import pre_process_using_xp, load_ribeiro_model
# from mim.model_wrapper import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from mim.cross_validation import ChronologicalSplit


class ESCT(Experiment, Enum):
    M_R1_CNN1 = Experiment(
        description='Predicting MACE-30 using single raw ECG in a simple '
                    '2-layer CNN.',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'dropout': 0.3,
                'filter_first': 32,
                'filter_last': 32,
                'kernel_first': 16,
                'kernel_last': 16,
                'pool_size': 16,
                'batch_norm': True,
                'dense': False,
                'downsample': False
            },
            'dense_size': 10,
            'dropout': 0.3
        },
        epochs=200,
        batch_size=64,
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 1e-4}
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0']
            },
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R1_CNN2 = M_R1_CNN1._replace(
        description='Try adjusting the final dense-layer size from 10 to 100.'
                    'Also downsamples the ECG first, to its original 500Hz.',
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'dropout': 0.3,
                'filter_first': 32,
                'filter_last': 32,
                'kernel_first': 16,
                'kernel_last': 16,
                'pool_size': 16,
                'batch_norm': True,
                'dense': False,
                'downsample': True
            },
            'dense_size': 100,
            'dropout': 0.3
        }
    )
    M_R1_CNN3 = M_R1_CNN2._replace(
        description='Add class-weights to the training, and also reduce '
                    'learning-rate when validation loss plateaus.',
        class_weight={0: 1, 1: 10},
        reduce_lr_on_plateau={
            'monitor': 'val_loss',
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-6
        },
    )
    M_R1_NOTCH_CNN3 = M_R1_CNN3._replace(
        description='Uses notch-filter and clipping to remove outliers and '
                    'baseline wander.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0']
            },
            "processing": {
                'notch-filter',
                'clip_outliers'
            }
        },
    )
    M_R1_CNN4 = M_R1_CNN2._replace(
        description='Increasing dropout, trying out a new lr schedule',
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'dropout': 0.5,
                'filter_first': 32,
                'filter_last': 32,
                'kernel_first': 16,
                'kernel_last': 16,
                'pool_size': 16,
                'batch_norm': True,
                'dense': False,
                'downsample': True
            },
            'dense_size': 100,
            'dropout': 0.5
        },
        class_weight={0: 1, 1: 10},
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [153*20, 153*40, 153*150],
                        'values': [1e-3, 1e-4, 1e-5, 1e-6],
                    }
                },
            }
        },
    )
    M_R1_CNN5 = M_R1_CNN4._replace(
        description='Adjusting pool-size and kernel-size.',
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'dropout': 0.5,
                'filter_first': 32,
                'filter_last': 32,
                'kernel_first': 32,
                'kernel_last': 16,
                'batch_norm': True,
                'dense': False,
                'downsample': True
            },
            'dense_size': 100,
            'dropout': 0.5
        },
    )
    M_R1_CNN6 = M_R1_CNN4._replace(
        description='Increasing dropout even further',
        model_kwargs={
            'cnn_kwargs': {
                'num_layers': 2,
                'dropout': 0.7,
                'filter_first': 32,
                'filter_last': 32,
                'kernel_first': 16,
                'kernel_last': 16,
                'pool_size': 16,
                'batch_norm': True,
                'dense': False,
                'downsample': True
            },
            'dense_size': 100,
            'dropout': 0.5
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [153 * 50, 153 * 100],
                        'values': [1e-3, 1e-4, 1e-5],
                    }
                },
            }
        },
    )
    M_R2_CNN4_NN1 = M_R1_CNN4._replace(
        description='Loads the pre-trained R1_CNN4 model and uses it as a '
                    'feature extractor for the two input ECGs. The model '
                    'itself is a simple feed-forward neural network with '
                    'a single hidden layer of size 100.',
        model=ffnn,
        model_kwargs={
            'dense_layers': [100],
            'dropout': 0.3
        },
        pre_processor=pre_process_using_xp,
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_CNN4',
            'commit': 'eb783dc3ab36f554b194cffe463919620d123496',
            'final_layer_index': -3
        },
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1']
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [153 * 100],
                        'values': [1e-4, 1e-5],
                    }
                },
            }
        },
    )
    M_R2_CNN4_RF1 = M_R2_CNN4_NN1._replace(
        description='Try Random Forest instead.',
        model=RandomForestClassifier,
        model_kwargs={
            'n_estimators': 1000,
        },
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_CNN4',
            'commit': 'eb783dc3ab36f554b194cffe463919620d123496',
            'final_layer_index': -3,
            'flatten': True  # We need this to avoid a dict input to RF
        },
        building_model_requires_development_data=False,
    )
    M_R1_RN1 = Experiment(
        description="Pretrained ResNet architecture from Ribeiro et al.",
        model=load_ribeiro_model,
        model_kwargs={
            'dense_layers': [],
            'dropout': 0.0,
            'freeze_resnet': False
        },
        epochs=200,
        batch_size=32,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [305 * 20, 305 * 100],
                        'values': [1e-3, 1e-4, 1e-5],
                    }
                },
            }
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            }
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R1_RN2 = M_R1_RN1._replace(
        description="Pretrained ResNet, with class-weights",
        class_weight={0: 1, 1: 10.7},
    )
    M_R1_RN3 = M_R1_RN1._replace(
        description='Pretrained ResNet, but adding a final dense 100 at the '
                    'end.',
        model_kwargs={
            'dense_layers': [100],
            'dropout': 0.0,
            'freeze_resnet': False
        },
    )
    M_R1_RN4 = M_R1_RN1._replace(
        description='Pretrained ResNet, but adding final dense 100 layer with '
                    'dropout at the end.',
        model_kwargs={
            'dense_layers': [100],
            'dropout': 0.3,
            'freeze_resnet': False
        },
    )
    M_R1_RN5 = M_R1_RN1._replace(
        description='Adjusting the learning-rate scheduler and reducing epoch '
                    'number. ',
        model_kwargs={
            'dense_layers': [100],
            'dropout': 0.0,
            'freeze_resnet': False
        },
        epochs=50,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [305 * 10, 305 * 20],
                        'values': [1e-3, 1e-4, 1e-5],
                    }
                },
            }
        },
    )
