from enum import Enum

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mim.experiments.experiments import Experiment
from mim.extractors.esc_trop import EscTrop
from mim.models.simple_nn import ecg_cnn, ffnn, logistic_regression
from mim.models.load import pre_process_using_xp, load_ribeiro_model
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

    # KERAS LOGISTIC REGRESSION, FLAT FEATURES:
    M_LR1_DT = Experiment(
        description='Logistic regression, mace vs dt',
        model=logistic_regression,
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt']
            },
        },
        epochs=300,
        batch_size=-1,
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1},
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_LR1_AGE = M_LR1_DT._replace(
        description='Logistic regression, mace vs dt',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt']
            },
        },
    )
    M_LR1_SEX = M_LR1_DT._replace(
        description='Logistic regression, mace vs sex',
        extractor_kwargs={
            "features": {
                'flat_features': ['male']
            },
        },
    )
    M_LR1_TNT = M_LR1_DT._replace(
        description='Logistic regression, mace vs tnt',
        extractor_kwargs={
            "features": {
                'flat_features': ['tnt_1']
            },
        },
    )
    M_LR1_DT_AGE = M_LR1_DT._replace(
        description='Logistic regression, mace vs dt + age',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age']
            },
        },
    )
    M_LR1_DT_AGE_SEX = M_LR1_DT._replace(
        description='Logistic regression, mace vs dt + age + sex',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male']
            },
        },
    )
    M_LR1_DT_AGE_SEX_TNT = M_LR1_DT._replace(
        description='Logistic regression, mace vs dt + age + sex + tnt',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
        },
    )

    # SKLEARN LOGISTIC REGRESSION, FLAT FEATURES:
    M_LR2_DT_AGE_SEX_TNT = Experiment(
        description='Scikit-learns logistic regression model, mace vs flat '
                    'features.',
        model=LogisticRegression,
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
        },
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
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

    # ABs CNN MODEL, 1 ECG + FLAT FEATURES
    M_R1_AB1 = Experiment(
        description='Predicting MACE-30 using only single raw ECG, '
                    'using the CNN architecture from Anders '
                    'Björkelund et al. ',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'downsample': False,
                'num_layers': 3,
                'dropouts': [0.0, 0.3, 0.0],
                'kernels': [64, 16, 16],
                'filters': [64, 16, 8],
                'weight_decays': [1e-4, 1e-3, 1e-4],
                'pool_sizes': [32, 4, 8],
                'batch_norm': False,
                'ffnn_kwargs': {
                    'sizes': [10],
                    'dropouts': [0.0],
                    'batch_norms': [False]
                },
            },
            'ecg_ffnn_kwargs': None,
            'flat_ffnn_kwargs': None,
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.5],
                'batch_norms': [False]
            }
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
            },
        },
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 3e-3}
        },
        epochs=200,
        batch_size=64,
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R1_AB1_DT_AGE_SEX_TNT = M_R1_AB1._replace(
        description='Predicting MACE-30 using single raw ECG and flat-'
                    'features, using the CNN architecture from Anders '
                    'Björkelund et al. ',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
        },
    )

    # LOGISTIC REGRESSION USING 1 ECG PROCESSED WITH CNN4 + FLAT FEATURES
    M_R1_CNN4_LR1_DT = Experiment(
        description='Pre-processing 1 input ECG with CNN4, into a single '
                    'scalar, then adding delta-t and feeding it into a '
                    'logistic-regression model.',
        model=logistic_regression,
        pre_processor=pre_process_using_xp,
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_CNN4',
            'commit': 'eb783dc3ab36f554b194cffe463919620d123496',
            'final_layer_index': -1
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt']
            },
        },
        epochs=300,
        batch_size=-1,
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1},
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R1_CNN4_LR1_AGE = M_R1_CNN4_LR1_DT._replace(
        description='ECG + Age',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['age']
            },
        },
    )
    M_R1_CNN4_LR1_SEX = M_R1_CNN4_LR1_DT._replace(
        description='ECG + Sex',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['male']
            },
        },
    )
    M_R1_CNN4_LR1_TNT = M_R1_CNN4_LR1_DT._replace(
        description='ECG + tnt',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['tnt_1']
            },
        },
    )
    M_R1_CNN4_LR1_DT_AGE = M_R1_CNN4_LR1_DT._replace(
        description='ECG + dt + age',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age']
            },
        },
    )
    M_R1_CNN4_LR1_DT_AGE_SEX = M_R1_CNN4_LR1_DT._replace(
        description='ECG + dt + age + sex',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male']
            },
        },
    )
    M_R1_CNN4_LR1_DT_AGE_SEX_TNT = M_R1_CNN4_LR1_DT._replace(
        description='ECG + age + sex + tnt',
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

    # LOGISTIC REGRESSION USING 1 ECG PROCESSED WITH RESNET + FLAT FEATURES
    M_R1_RN5_LR1_DT_AGE_SEX_TNT = Experiment(
        description='Loads the pre-trained R1_RN5 model and uses it as a '
                    'feature extractor for the input ECG, giving only the '
                    'final scalar as output for each ECG. Add the flat-'
                    'features and plug it all into a logistic regression '
                    'model.',
        model=logistic_regression,
        pre_processor=pre_process_using_xp,
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_RN5',
            'commit': '61fb8038d91ee119a1a889c3c86b27931f1f57b5',
            'which': 'last',
            'final_layer_index': -1
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
        epochs=300,
        batch_size=-1,
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1},
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
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

    # LOGISTIC REGRESSION USING 2 ECGs PROCESSED WITH CNN4 + FLAT FEATURES
    M_R2_CNN4_LR1_DT = Experiment(
        description='Logistic regression, 2 ECGs + dt vs MACE 30',
        model=logistic_regression,
        pre_processor=pre_process_using_xp,
        pre_processor_kwargs={
            'xp_name': 'ESCT/M_R1_CNN4',
            'commit': 'eb783dc3ab36f554b194cffe463919620d123496',
            'final_layer_index': -1
        },
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt']
            },
        },
        epochs=300,
        batch_size=-1,
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1},
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    M_R2_CNN4_LR1_AGE = M_R2_CNN4_LR1_DT._replace(
        description='Logistic regression, 2 ECGs + age vs MACE 30',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['age']
            },
        },
    )
    M_R2_CNN4_LR1_SEX = M_R2_CNN4_LR1_DT._replace(
        description='Logistic regression, 2 ECGs + sex vs MACE 30',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['male']
            },
        },
    )
    M_R2_CNN4_LR1_TNT = M_R2_CNN4_LR1_DT._replace(
        description='Logistic regression, 2 ECGs + tnt vs MACE 30',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['tnt_1']
            },
        },
    )
    M_R2_CNN4_LR1_DT_AGE = M_R2_CNN4_LR1_DT._replace(
        description='Logistic regression, 2 ECGs + dt + age vs MACE 30',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt', 'age']
            },
        },
    )
    M_R2_CNN4_LR1_DT_AGE_SEX = M_R2_CNN4_LR1_DT._replace(
        description='Logistic regression, 2 ECGs + dt + age + sex vs MACE 30',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt', 'age', 'male']
            },
        },
    )
    M_R2_CNN4_LR1_DT_AGE_SEX_TNT = M_R2_CNN4_LR1_DT._replace(
        description='Logistic regression, 2 ECGs + all flat features vs '
                    'MACE 30',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0', 'ecg_1'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
        },
    )

    # LOGISTIC REGRESSION USING 2 ECGs PROCESSED WITH RESNET + FLAT FEATURES
    M_R2_RN5_LR1_DT_AGE_SEX_TNT = M_R1_RN5_LR1_DT_AGE_SEX_TNT._replace(
        description='Process both input ECGs with the pre-trained ResNet, '
                    'using only the predictions for each as input, together '
                    'with the flat-features.',
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

    # FFNN USING 2 ECGs PROCESSED WITH RESNET + FLAT FEATURES
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

    # AMI-30
    AMI_LR1_TNT = Experiment(
        description='Logistic regression, ami vs tnt',
        model=logistic_regression,
        extractor=EscTrop,
        extractor_kwargs={
            "features": {
                'flat_features': ['tnt_1']
            },
            'labels': {
                'target': 'ami30'
            }
        },
        epochs=300,
        batch_size=-1,
        optimizer={
            'name': SGD,
            'kwargs': {'learning_rate': 1},
        },
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    AMI_LR1_DT_AGE_SEX_TNT = AMI_LR1_TNT._replace(
        description='Logistic regression, ami vs tnt + dt + age + sex',
        extractor_kwargs={
            "features": {
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
            'labels': {
                'target': 'ami30'
            }
        }
    )

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
    AMI_R1_CNN4 = Experiment(
        description='Predicting AMI-30 with CNN4 and only 1 ECG input.',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'downsample': True,
                'num_layers': 2,
                'dropout': 0.5,
                'filter_first': 32,
                'filter_last': 32,
                'kernel_first': 16,
                'kernel_last': 16,
                'pool_size': 16,
                'batch_norm': True,
                'ffnn_kwargs': None,
            },
            'ecg_ffnn_kwargs': {
                'sizes': [100],
                'dropouts': [0.5],
                'batch_norms': [False]
            },
            'flat_ffnn_kwargs': None,
            'final_fnn_kwargs': None
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
        class_weight={0: 1, 1: 10.7},
        epochs=200,
        batch_size=64,
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )

    AMI_R1_CNN4_TNT = AMI_R1_CNN4._replace(
        description='Predicting AMI-30 with CNN4 and 1 ECG input + TnT',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['tnt_1']
            },
            'labels': {
                'target': 'ami30'
            }
        },
    )
    AMI_R1_CNN4_DT_AGE_SEX_TNT = AMI_R1_CNN4._replace(
        description='Predicting AMI-30 with CNN4 and 1 ECG input + flat-'
                    'features',
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
            'labels': {
                'target': 'ami30'
            }
        },
    )

    # TODO: Once AMI_R1_CNN4 is done
    # AMI_R1_CNN4_LR1_TNT
    # AMI_R1_CNN4_LR1_DT_AGE_SEX_TNT
    #
    # AMI_R2_CNN4_LR1_TNT
    # AMI_R2_CNN4_LR1_DT_AGE_SEX_TNT

    AMI_R1_AB1 = Experiment(
        description='Predicting AMI-30 using Björkelund et al, except only '
                    '1 ECG input and nothing else.',
        model=ecg_cnn,
        model_kwargs={
            'cnn_kwargs': {
                'downsample': False,
                'num_layers': 3,
                'dropouts': [0.0, 0.3, 0.0],
                'kernels': [64, 16, 16],
                'filters': [64, 16, 8],
                'weight_decays': [1e-4, 1e-3, 1e-4],
                'pool_sizes': [32, 4, 8],
                'batch_norm': False,
                'ffnn_kwargs': {
                    'sizes': [10],
                    'dropouts': [0.0],
                    'batch_norms': [False]
                },
            },
            'final_ffnn_kwargs': {
                'sizes': [10],
                'dropouts': [0.5],
                'batch_norms': [False]
            }
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
        optimizer={
            'name': Adam,
            'kwargs': {'learning_rate': 3e-3}
        },
        epochs=200,
        batch_size=64,
        building_model_requires_development_data=True,
        cv=ChronologicalSplit,
        cv_kwargs={'test_size': 1 / 3},
        scoring=roc_auc_score,
    )
    AMI_R1_AB1_DT_AGE_SEX_TNT = AMI_R1_AB1._replace(
        extractor_kwargs={
            "features": {
                'ecg_mode': 'raw',
                'ecgs': ['ecg_0'],
                'flat_features': ['log_dt', 'age', 'male', 'tnt_1']
            },
            'labels': {
                'target': 'ami30'
            }
        },
    )

    AMI_R1_RN5 = Experiment(
        description="Pretrained ResNet architecture from Ribeiro et al, "
                    "predicting AMI-30.",
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

    # TODO once AMI_R1_RN5 is done
    # AMI_R1_RN5_NN1_TNT
    # AMI_R1_RN5_NN1_DT_AGE_SEX_TNT
    #
    # AMI_R2_RN5_NN1_TNT
    # AMI_R2_RN5_NN1_DT_AGE_SEX_TNT
