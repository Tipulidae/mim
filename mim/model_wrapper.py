# -*- coding: utf-8 -*-

import os
from enum import Enum

import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
    ReduceLROnPlateau
)

from mim.util.logs import get_logger
from mim.util.util import keras_model_summary_as_string

log = get_logger('Model Wrapper')


class ModelTypes(Enum):
    CLASSIFIER = 0
    REGRESSOR = 1

    @property
    def is_classification(self):
        return self in [ModelTypes.CLASSIFIER]

    @property
    def is_regression(self):
        return self in [ModelTypes.REGRESSOR]


class Model:
    model_type = ModelTypes.CLASSIFIER

    def __init__(
            self,
            model,
            can_use_tf_dataset=False,
    ):
        self.model = model
        self.can_use_tf_dataset = can_use_tf_dataset

    def predict(self, x):
        result = {}
        if self.can_use_tf_dataset:
            x = x.as_dataset()
        else:
            x = x.as_flat_numpy()
        prediction = self._prediction(x)

        if self.only_last_prediction_column_is_used:
            prediction = prediction[:, 1]

        result['prediction'] = pd.DataFrame(prediction)
        return result

    def fit(self, data, validation_data=None, **kwargs):
        if self.can_use_tf_dataset:
            train = prepare_dataset(data, prefetch=3, **kwargs)
            val = prepare_dataset(validation_data, prefetch=3, **kwargs)
            return self.model.fit(train, validation_data=val, **kwargs).history
        else:
            x = data['x'].as_flat_numpy()
            y = data['y'].as_numpy().ravel()
            return self.model.fit(x, y)

    @property
    def summary(self):
        return None

    @property
    def only_last_prediction_column_is_used(self):
        return (self.model_type is ModelTypes.CLASSIFIER and
                len(self.model.classes_) == 2)

    def _prediction(self, x):
        return self.model.predict_proba(x)


def _numpy(array):
    if isinstance(array, (pd.DataFrame, pd.Series)):
        return array.values
    else:
        return array


class RandomForestClassifier(Model):
    def __init__(self, *args, random_state=123, **kwargs):
        super().__init__(
            ensemble.RandomForestClassifier(
                *args,
                random_state=random_state,
                **kwargs
            )
        )


class ExtraTreesClassifier(Model):
    def __init__(self, *args, random_state=124, **kwargs):
        super().__init__(
            ensemble.ExtraTreesClassifier(
                *args,
                random_state=random_state,
                **kwargs
            )
        )


class GradientBoostingClassifier(Model):
    def __init__(self, *args, random_state=125, **kwargs):
        super().__init__(
            ensemble.GradientBoostingClassifier(
                *args,
                random_state=random_state,
                **kwargs
            )
        )


class LogisticRegression(Model):
    def __init__(self, *args, random_state=125, **kwargs):
        super().__init__(
            linear_model.LogisticRegression(
                *args,
                random_state=random_state, **kwargs
            )
        )


class LinearRegression(Model):
    model_type = ModelTypes.REGRESSOR

    def __init__(self, *args, **kwargs):
        super().__init__(
            linear_model.LinearRegression(
                *args,
                **kwargs
            )
        )

    def _prediction(self, x):
        return self.model.predict(x)


class RandomForestRegressor(Model):
    model_type = ModelTypes.REGRESSOR

    def __init__(self, *args, random_state=126, **kwargs):
        super().__init__(
            ensemble.RandomForestRegressor(
                *args,
                random_state=random_state,
                **kwargs
            )
        )

    def _prediction(self, x):
        return self.model.predict(x)


class NullModel:
    classes_ = []

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, x, y):
        pass

    def predict_proba(self, x):
        return None


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return
        optimizer = self.model.optimizer
        # This is a bit daft, but the best (only) way I found that works.
        if isinstance(optimizer.learning_rate,
                      tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = optimizer._decayed_lr('float32').numpy()
            logs['learning_rate'] = lr


class KerasWrapper(Model):
    def __init__(
            self,
            model: tf.keras.Model,
            batch_size=16,
            epochs=2,
            initial_epoch=0,
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=None,
            ignore_callbacks=False,
            checkpoint_path=None,
            skip_compile=False,
            tensorboard_path=None,
            exp_base_path=None,
            class_weight=None,
            reduce_lr_on_plateau=None
    ):
        super().__init__(model, can_use_tf_dataset=True)
        if not skip_compile:
            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )
        self.checkpoint_path = checkpoint_path
        self.tensorboard_path = tensorboard_path
        self.exp_base_path = exp_base_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_epoch = initial_epoch
        self.ignore_callbacks = ignore_callbacks
        self.class_weight = class_weight
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        log.info("\n\n" + keras_model_summary_as_string(model))

    def fit(self, data, validation_data=None, split_number=None, **kwargs):
        keras.utils.plot_model(self.model,
                               os.path.join(self.exp_base_path,
                                            "network-graph.png"),
                               True, True)
        if self.ignore_callbacks:
            callbacks = None
        else:
            if split_number is None:
                split_folder = ""
            else:
                split_folder = f'split_{split_number}'

            checkpoint = os.path.join(self.checkpoint_path, split_folder)
            tensorboard = os.path.join(self.tensorboard_path, split_folder)

            callbacks = [
                ModelCheckpoint(
                    filepath=os.path.join(checkpoint, 'last.ckpt')
                ),
                ModelCheckpoint(
                    filepath=os.path.join(checkpoint, 'best.ckpt'),
                    save_best_only=True
                ),
                TensorBoard(log_dir=tensorboard),
                LearningRateLogger()
            ]

            if self.reduce_lr_on_plateau is not None:
                callbacks.append(
                    ReduceLROnPlateau(**self.reduce_lr_on_plateau)
                )
        if self.batch_size < 0:
            self.batch_size = len(data)
        return super().fit(
            data,
            validation_data=validation_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            initial_epoch=self.initial_epoch,
            callbacks=callbacks,
            class_weight=self.class_weight,
            **kwargs
        )

    @property
    def summary(self):
        return keras_model_summary_as_string(self.model)

    @property
    def only_last_prediction_column_is_used(self):
        return False

    def _prediction(self, x):
        return self.model.predict(x.batch(self.batch_size))


def prepare_dataset(data, batch_size=1, prefetch=None, **kwargs):
    # If the data doesn't fit in memory, we can't use tf shuffling. Instead,
    # we shuffle (lazily) the generator that becomes the dataset. This does
    # nothing if the data does fit in memory.
    x = data['x'].as_dataset(shuffle=True)
    y = data['y'].as_dataset(shuffle=True)
    fixed_data = tf.data.Dataset.zip((x, y))

    # If the data _does_ fit in memory, we can use the tf shuffling instead.
    # This would be bad if data doesn't fit in memory though, because tf will
    # load the entire dataset in memory before shuffling.
    if data.fits_in_memory:
        fixed_data = fixed_data.shuffle(
            len(data),
            reshuffle_each_iteration=True  # Best to be explicit I suppose
        )

    fixed_data = fixed_data.batch(batch_size)

    if prefetch:
        fixed_data = fixed_data.prefetch(prefetch)

    return fixed_data
