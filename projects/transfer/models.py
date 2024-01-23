import random
from collections import OrderedDict

import numpy as np
import torch
from tensorflow import keras
from keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    Activation,
    Flatten,
    Dense,
    Concatenate,
    Dropout
)

from mim.models.load import (
    load_model_from_experiment_result, load_model_from_experiment_result_pt,
    load_ribeiro_model
)
from mim.models.util import (
    cnn_helper, mlp_helper, ResidualUnit, ResidualUnitV2
)
from .xresnet import XResNet1d, init_cnn


def cnn(train, validation=None, cnn_kwargs=None, ffnn_kwargs=None):
    inp = {
        key: Input(shape=value, name=key)
        for key, value in train.feature_tensor_shape.items()
    }
    x = cnn_helper(inp['ecg'], **cnn_kwargs)
    if ffnn_kwargs:
        x = mlp_helper(x, **ffnn_kwargs)

    output = _final_layer(x, train.target_columns)
    return keras.Model(inp, output)


def pretrained(train, validation=None, from_xp=None, ecg_mlp_kwargs=None,
               flat_mlp_kwargs=None, final_mlp_kwargs=None, ecg_dropout=0.0):
    inp = {}
    if 'flat_features' in train.feature_tensor_shape:
        inp['flat_features'] = Input(
            shape=train.feature_tensor_shape['flat_features'],
        )
    ecg_inp, x = load_model_from_experiment_result(**from_xp)
    inp['ecg'] = ecg_inp
    x = Dropout(ecg_dropout)(x)

    if ecg_mlp_kwargs is not None:
        x = mlp_helper(x, **ecg_mlp_kwargs)

    if flat_mlp_kwargs is not None:
        flat_features = mlp_helper(inp['flat_features'], **flat_mlp_kwargs)
        x = Concatenate()([x, flat_features])

    if final_mlp_kwargs is not None:
        x = mlp_helper(x, **final_mlp_kwargs)
    output = Dense(units=1, activation='sigmoid', kernel_regularizer='l2')(x)
    return keras.Model(inp, output)


def pretrained_parallel(
        train, validation=None, from_xp1=None, from_xp2=None,
        ecg_mlp_kwargs=None, flat_mlp_kwargs=None, final_mlp_kwargs=None,
        ecg_dropout=0.0):
    inp = {}
    if 'flat_features' in train.feature_tensor_shape:
        inp['flat_features'] = Input(
            shape=train.feature_tensor_shape['flat_features'],
        )
    ecg_inp1, x1 = load_model_from_experiment_result(**from_xp1)
    ecg_inp2, x2 = load_model_from_experiment_result(**from_xp2)
    inp['ecg'] = Input(shape=(4096, 8), dtype=np.float16, name='signal')

    m1 = keras.Model(ecg_inp1, x1, name='model1')
    m2 = keras.Model(ecg_inp2, x2, name='model2')

    x1 = m1(inp['ecg'])
    x2 = m2(inp['ecg'])
    x = Concatenate()([x1, x2])
    x = Dropout(ecg_dropout)(x)

    if ecg_mlp_kwargs is not None:
        x = mlp_helper(x, **ecg_mlp_kwargs)

    if flat_mlp_kwargs is not None:
        flat_features = mlp_helper(inp['flat_features'], **flat_mlp_kwargs)
        x = Concatenate()([x, flat_features])

    if final_mlp_kwargs is not None:
        x = mlp_helper(x, **final_mlp_kwargs)
    output = Dense(units=1, activation='sigmoid', kernel_regularizer='l2')(x)
    return keras.Model(inp, output)


def ribeiros_resnet(train, validation=None, final_mlp_kwargs=None):
    inp, x = load_ribeiro_model(freeze_resnet=True, suffix='_rn')
    if final_mlp_kwargs is not None:
        x = mlp_helper(x, **final_mlp_kwargs)

    output = Dense(units=1, activation='sigmoid', kernel_regularizer='l2')(x)
    return keras.Model({'ecg': inp}, output)


def resnet_v1(train, validation=None, residual_kwargs=None,
              flat_mlp_kwargs=None, ecg_mlp_kwargs=None,
              final_mlp_kwargs=None):
    if residual_kwargs is None:
        residual_kwargs = {}

    inp = {}
    if 'flat_features' in train.feature_tensor_shape:
        inp['flat_features'] = Input(
            shape=train.feature_tensor_shape['flat_features'],
        )
    if 'ecg' in train.feature_tensor_shape:
        inp['ecg'] = Input(shape=(4096, 8), dtype=np.float16, name='signal')

    x = Conv1D(64, 17, padding='same', use_bias=False,
               kernel_initializer='he_normal')(inp['ecg'])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x, y = ResidualUnit(1024, 128, **residual_kwargs)([x, x])
    x, y = ResidualUnit(256, 196, **residual_kwargs)([x, y])
    x, y = ResidualUnit(64, 256, **residual_kwargs)([x, y])
    x, _ = ResidualUnit(16, 320, **residual_kwargs)([x, y])
    x = Flatten()(x)

    if ecg_mlp_kwargs is not None:
        x = mlp_helper(x, **ecg_mlp_kwargs)
    if 'flat_features' in inp:
        flat_features = inp['flat_features']
        if flat_mlp_kwargs is not None:
            flat_features = mlp_helper(flat_features, **flat_mlp_kwargs)
        x = Concatenate()([x, flat_features])
    if final_mlp_kwargs is not None:
        x = mlp_helper(x, **final_mlp_kwargs)

    output = _final_layer(x, train.target_columns)
    model = keras.Model(inp, output)
    return model


def resnet_v2(
        train, validation=None, filters=64, residual_kwargs=None,
        flat_mlp_kwargs=None, ecg_mlp_kwargs=None, final_mlp_kwargs=None
):
    if residual_kwargs is None:
        residual_kwargs = {}

    inp = {}
    if 'flat_features' in train.feature_tensor_shape:
        inp['flat_features'] = Input(
            shape=train.feature_tensor_shape['flat_features'],
        )
    if 'ecg' in train.feature_tensor_shape:
        inp['ecg'] = Input(shape=(4096, 8), dtype=np.float16, name='signal')

    x = Conv1D(64, 17, padding='same', use_bias=False,
               kernel_initializer='he_normal')(inp['ecg'])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x, y = ResidualUnitV2(2048, filters, **residual_kwargs)([x, x])
    x, y = ResidualUnitV2(2048, filters, **residual_kwargs)([x, y])
    x, y = ResidualUnitV2(1024, filters, **residual_kwargs)([x, y])

    x, y = ResidualUnitV2(512, 2*filters, **residual_kwargs)([x, y])
    x, y = ResidualUnitV2(512, 2*filters, **residual_kwargs)([x, y])
    x, y = ResidualUnitV2(256, 2*filters, **residual_kwargs)([x, y])

    x, y = ResidualUnitV2(128, 4*filters, **residual_kwargs)([x, y])
    x, y = ResidualUnitV2(128, 4*filters, **residual_kwargs)([x, y])
    x, y = ResidualUnitV2(64, 4*filters, **residual_kwargs)([x, y])

    x, y = ResidualUnitV2(32, 8*filters, **residual_kwargs)([x, y])
    x, y = ResidualUnitV2(32, 8*filters, **residual_kwargs)([x, y])
    x, _ = ResidualUnitV2(16, 8*filters, **residual_kwargs)([x, y])

    x = Flatten()(x)

    if ecg_mlp_kwargs is not None:
        x = mlp_helper(x, **ecg_mlp_kwargs)
    if 'flat_features' in inp:
        flat_features = inp['flat_features']
        if flat_mlp_kwargs is not None:
            flat_features = mlp_helper(flat_features, **flat_mlp_kwargs)
        x = Concatenate()([x, flat_features])
    if final_mlp_kwargs is not None:
        x = mlp_helper(x, **final_mlp_kwargs)

    output = _final_layer(x, train.target_columns)
    model = keras.Model(inp, output)
    return model


def _final_layer(x, target_columns):
    output_layers = []
    for name in target_columns:
        if name not in ['age', 'sex', 'ami']:
            raise ValueError(f"Target column must be either age, sex, or ami "
                             f"but it was {name}.")
        binary_target = name in ['sex', 'ami']
        output_layers.append(
            Dense(
                units=1,
                activation='sigmoid' if binary_target else None,
                kernel_initializer='he_normal',
                kernel_regularizer='l2',
                name=name
            )(x)
        )

    if len(output_layers) > 1:
        output = output_layers
    else:
        output = output_layers[0]

    return output


def mlp(train, validation=None, mlp_kwargs=None):
    inp = Input(shape=train.feature_tensor_shape)
    x = mlp_helper(inp, **mlp_kwargs)
    output = Dense(units=1, activation='sigmoid', kernel_regularizer='l2')(x)
    return keras.Model(inp, output)


def simple_mlp_tf(validation=None, **kwargs):
    inp = Input(100)
    x = Dense(50, activation='relu')(inp)
    output = Dense(1, activation='sigmoid', kernel_regularizer='l2')(x)
    return keras.Model(inp, output)


def simple_mlp_pt(validation=None, **kwargs):
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1),
        torch.nn.Sigmoid(),
    )
    return model


def _make_input(shape):
    if isinstance(shape, dict):
        return {key: _make_input(value) for key, value in shape.items()}
    else:
        return Input(shape=shape)


class FixInput(torch.nn.Module):
    def forward(self, x):
        return torch.transpose(x['ecg'], 1, 2)


class MultiClassifier(torch.nn.Module):
    def __init__(self, target_columns):
        super().__init__()
        self.classifiers = torch.nn.ModuleDict({
            target: torch.nn.Sequential(
                torch.nn.BatchNorm1d(512),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, 1),
            ) for target in target_columns
        })
        self.targets = target_columns

    def forward(self, input):
        return torch.cat(
            [self.classifiers[target](input) for target in self.targets],
            dim=1
        )


class AugmentECG(torch.nn.Sequential):
    def __init__(self, *args, random_seed=123, mode='batch', reduction='max',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_mode = mode == 'batch'
        if mode not in ['batch', 'sample']:
            raise ValueError(
                f"mode should be 'batch' or 'sample', was {mode}")
        if reduction not in ['max', 'mean']:
            raise ValueError(
                f"reduction should be 'max' or 'mean', was {reduction}")

        # torch.max returns a tuple of the max and the corresponding indices,
        # while torch.amax only returns the maxima, which is what I want here.
        self.reduction = torch.mean if reduction == 'mean' else torch.amax
        self.random_generator = random.Random(random_seed)

    def forward(self, input):
        input = torch.transpose(input['ecg'], 1, 2)
        resolution = input.shape[-1]
        window_size = resolution // 4
        eval_steps = 10
        batch_size = input.shape[0]
        if self.training:
            random_ecgs = []
            if self.batch_mode:
                idx = self.random_generator.randint(0, window_size * 3)
                random_slice = input[:, :, idx:idx + window_size]
            else:
                for i in range(batch_size):
                    idx = self.random_generator.randint(0, window_size * 3)
                    random_slice = input[i, :, idx:idx + window_size]
                    random_ecgs.append(random_slice)
                random_slice = torch.stack(random_ecgs)
            return super().forward(random_slice)
        else:
            slices = [
                input[:, :, i:i + window_size]
                for i in np.linspace(
                    0, resolution - window_size, eval_steps).astype(int)
            ]
            preds = []
            for slice in slices:
                preds.append(super().forward(slice))

            predictions = torch.stack(preds)
            return self.reduction(predictions, 0)


def xrn50(train, validation=None, initial_bn=False,
          augmentation=None, **kwargs):
    ecg_shape = train.feature_tensor_shape['ecg']
    sample_rate = ecg_shape[0] // 10
    num_leads = ecg_shape[1]
    resnet = XResNet1d(
        expansion=4,
        blocks=[3, 4, 6, 3],
        inp_dim=num_leads,
        out_dim=train.output_size,
        sample_rate=sample_rate
    )
    # if len(train.target_columns) > 1:
    #     resnet.head.classifier = MultiClassifier(train.target_columns)
    #     init_cnn(resnet.head.classifier)

    layers = OrderedDict()
    if augmentation is None:
        layers['input'] = FixInput()
    if initial_bn:
        bn = torch.nn.BatchNorm1d(8)
        bn.weight.data.fill_(1.0)
        layers['normalization'] = bn

    layers['resnet'] = resnet

    if augmentation is None:
        model = torch.nn.Sequential(layers)
    else:
        model = AugmentECG(layers, **augmentation)

    return model


def pretrained_pt(*args, from_xp=None, **kwargs):
    model = load_model_from_experiment_result_pt(**from_xp)

    new_classifier_head = torch.nn.Sequential(
        torch.nn.Linear(in_features=512, out_features=100, bias=True),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(in_features=100, out_features=1, bias=True)
    )
    init_cnn(new_classifier_head)
    model.resnet.head.classifier = new_classifier_head
    return model
