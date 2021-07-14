from enum import Enum

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from sklearn.metrics import roc_auc_score

from mim.experiments.experiments import Experiment
from mim.extractors.esc_trop import EscTrop
from mim.models.simple_nn import ecg_cnn, ffnn
from mim.models.load import pre_process_using_xp, load_ribeiro_model
from sklearn.ensemble import RandomForestClassifier
from mim.cross_validation import ChronologicalSplit


class ESCT(Experiment, Enum):
    # RANDOM FOREST, FLAT FEATURES:
    M_RF1_DT = Experiment(
        description='Predicting MACE with Random Forest, using only the '
                    'time since last ECG.',
        model=RandomForestClassifier,
        model_kwargs={
            'n_estimators': 1000,
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt']
            },
        },
        building_model_requires_development_data=False,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_RF1_AGE = M_RF1_DT._replace(
        description='Predicting MACE with Random Forest, using only the '
                    'patient age.',
        extractor_kwargs={
            "features": {
                'flat_features': ['age']
            },
        },
    )
    M_RF1_SEX = M_RF1_DT._replace(
        description='Predicting MACE with Random Forest, using only the '
                    'patient sex.',
        extractor_kwargs={
            "features": {
                'flat_features': ['male']
            },
        },
    )
    M_RF1_TNT = M_RF1_DT._replace(
        description='Predicting MACE with Random Forest, using only the '
                    'first TnT lab measurement.',
        extractor_kwargs={
            "features": {
                'flat_features': ['tnt_1']
            },
        },
    )
    M_RF1_DT_AGE = M_RF1_DT._replace(
        description='Predicting MACE with Random Forest, using only the '
                    'time since last ECG and patient age.',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age']
            },
        },
    )
    M_RF1_DT_AGE_SEX = M_RF1_DT._replace(
        description='Predicting MACE with Random Forest, using only the '
                    'time since last ECG, patient age and sex.',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male']
            },
        },
    )
    M_RF1_DT_AGE_SEX_TNT = M_RF1_DT._replace(
        description='Predicting MACE with Random Forest, using only the '
                    'time since last ECG, age, sex and first TnT measurement.',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
        },
    )

    # SINGLE RAW ECG, CNN VARIATIONS:
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

    # CNN4, 1 ECG + FLAT FEATURES
    M_R1_CNN4_DT = M_R1_CNN4._replace(
        description='Adds the (logarithm of the) time since last ECG.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt']
            },
        },
    )
    M_R1_CNN4_AGE = M_R1_CNN4._replace(
        description='Adds age.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['age']
            },
        },
    )
    M_R1_CNN4_SEX = M_R1_CNN4._replace(
        description='Adds sex (1 = male, 0 = female).',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['male']
            },
        },
    )
    M_R1_CNN4_TNT = M_R1_CNN4._replace(
        description='Adds the first TnT lab measurement.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['tnt_1']
            },
        },
    )
    M_R1_CNN4b_TNT = M_R1_CNN4_TNT._replace(
        description='Adjusts the learning-rate schedule. Also, what if we '
                    'skip the class-weights?',
        epochs=100,
        class_weight=None,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [153 * 5],
                        'values': [1e-3, 1e-4],
                    }
                },
            }
        },
    )
    M_R1_CNN4_DT_AGE = M_R1_CNN4._replace(
        description='Adds time since last ECG and age.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age']
            },
        },
    )
    M_R1_CNN4_DT_AGE_SEX = M_R1_CNN4._replace(
        description='Adds time since last ECG, age and sex.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male']
            },
        },
    )
    M_R1_CNN4_DT_AGE_SEX_TNT = M_R1_CNN4._replace(
        description='Adds time since last ECG, age, sex and TnT.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
        },
    )

    # SINGLE RAW ECG, RESNET VARIATIONS
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

    # FFNN USING 1 ECG PROCESSED WITH RESNET + FLAT FEATURES
    M_R1_RN5_NN1_DT_AGE_SEX_TNT = Experiment(
        description='Loads the pre-trained R1_RN5 model and uses it as a '
                    'feature extractor for the input ECG. Feed this into a '
                    'dense-100 layer, then concatenate some flat-features '
                    'before the final sigmoid layer.',
        model=ffnn,
        model_kwargs={
            'dense_layers': [100],
            'dropout': 0.3
        },
        pre_processor=pre_process_using_xp,
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_RN5',
            'commit': '61fb8038d91ee119a1a889c3c86b27931f1f57b5',
            'which': 'last',
            'final_layer_index': -3
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            }
        },
        epochs=200,
        batch_size=32,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [305 * 100],
                        'values': [1e-3, 1e-4],
                    }
                },
            }
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R1_RN5_NN2_DT_AGE_SEX_TNT = M_R1_RN5_NN1_DT_AGE_SEX_TNT._replace(
        description='Reducing the size of the NN to just a Dense-10. Also, '
                    'no dropout for now.',
        model_kwargs={
            'dense_layers': [10],
            'dropout': 0.0
        },
        epochs=100,
        batch_size=32,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [305 * 50],
                        'values': [1e-3, 1e-4],
                    }
                },
            }
        },
    )
    M_R1_RN5_NN3_DT_AGE_SEX_TNT = M_R1_RN5_NN1_DT_AGE_SEX_TNT._replace(
        description='Neural network with 100 -> 1 dense layers.',
        model_kwargs={
            'dense_layers': [100, 1],
            'dropout': 0.0
        },
        epochs=100,
        batch_size=32,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [305 * 50],
                        'values': [1e-3, 1e-4],
                    }
                },
            }
        },
    )

    # FFNN USING 2 ECGs PROCESSED WITH CNN4 + FLAT FEATURES
    M_R2_CNN4_NN1 = Experiment(
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
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1']
            },
        },
        class_weight={0: 1, 1: 10},
        epochs=200,
        batch_size=64,
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
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R2_CNN4_NN1_DT = M_R2_CNN4_NN1._replace(
        description='Use pretrained CNN as feature extractor for ECGs. Feed '
                    'into dense-100 layer, then concatenate time since last '
                    'ecg.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt']
            },
        },
    )
    M_R2_CNN4_NN1_AGE = M_R2_CNN4_NN1._replace(
        description='Use pretrained CNN as feature extractor for ECGs. Feed '
                    'into dense-100 layer, then concatenate age.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['age']
            },
        },
    )
    M_R2_CNN4_NN1_SEX = M_R2_CNN4_NN1._replace(
        description='Use pretrained CNN as feature extractor for ECGs. Feed '
                    'into dense-100 layer, then concatenate sex.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['male']
            },
        },
    )
    M_R2_CNN4_NN1_TNT = M_R2_CNN4_NN1._replace(
        description='Use pretrained CNN as feature extractor for ECGs. Feed '
                    'into dense-100 layer, then concatenate tnt.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['tnt_1']
            },
        },
    )
    M_R2_CNN4_NN1_DT_AGE = M_R2_CNN4_NN1._replace(
        description='Use pretrained CNN as feature extractor for ECGs. Feed '
                    'into dense-100 layer, then concatenate time since last '
                    'ecg and age.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt', 'age']
            },
        },
    )
    M_R2_CNN4_NN1_DT_AGE_SEX = M_R2_CNN4_NN1._replace(
        description='Use pretrained CNN as feature extractor for ECGs. Feed '
                    'into dense-100 layer, then concatenate time since last '
                    'ecg, age and sex',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt', 'age', 'male']
            },
        },
    )
    M_R2_CNN4_NN1_DT_AGE_SEX_TNT = M_R2_CNN4_NN1._replace(
        description='Use pretrained CNN as feature extractor for ECGs. Feed '
                    'into dense-100 layer, then concatenate time since last '
                    'ecg, age, sex and tnt.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
        },
    )

    # RF USING 2 ECGs PROCESSED WITH CNN4
    M_R2_CNN4_RF1 = M_R2_CNN4_NN1._replace(
        description='Try Random Forest instead.',
        model=RandomForestClassifier,
        model_kwargs={
            'n_estimators': 1000,
        },
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_CNN4',
            'commit': 'eb783dc3ab36f554b194cffe463919620d123496',
            'final_layer_index': -3
        },
        building_model_requires_development_data=False,
    )

    # FFNN USING 1 ECG PROCESSED WITH RESNET + FLAT FEATURES
    M_R2_RN5_NN1 = Experiment(
        description='Loads the pre-trained R1_RN5 model and uses it as a '
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
            'xp_name': 'ESCT/M_R1_RN5',
            'commit': '61fb8038d91ee119a1a889c3c86b27931f1f57b5',
            'which': 'last',
            'final_layer_index': -3
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            }
        },
        epochs=200,
        batch_size=32,
        optimizer={
            'name': Adam,
            'kwargs': {
                'learning_rate': {
                    'scheduler': PiecewiseConstantDecay,
                    'scheduler_kwargs': {
                        'boundaries': [305 * 100],
                        'values': [1e-3, 1e-4],
                    }
                },
            }
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R2_RN5_NN1_DT_AGE_SEX_TNT = M_R2_RN5_NN1._replace(
        description='Use two ECGs this time.',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
            'processing': {
                'scale': 1000,
                'ribeiro': True
            }
        },
    )

    # RF USING 2 ECGs PROCESSED WITH RESNET
    M_R2_RN5_RF1 = M_R2_RN5_NN1._replace(
        description='Pre-process the two input ECGs using pretrained ResNet, '
                    'concatenate the result and feed it into a Random Forest '
                    'classifier.',
        model=RandomForestClassifier,
        model_kwargs={
            'n_estimators': 1000,
        },
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_RN5',
            'commit': '61fb8038d91ee119a1a889c3c86b27931f1f57b5',
            'which': 'last',
            'final_layer_index': -3
        },
        building_model_requires_development_data=False,
    )

    # LOOKING AT AMI30 INSTEAD OF MACE
    AMI_R1_CNN2 = Experiment(
        description='Predicting AMI-30 using single raw ECG in a simple '
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
                'downsample': True
            },
            'dense_size': 100,
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
            'labels': {
                'target': 'ami30'
            }
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
