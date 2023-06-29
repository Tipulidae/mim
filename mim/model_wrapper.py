# -*- coding: utf-8 -*-

import os
from pathlib import Path
from enum import Enum
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard, ReduceLROnPlateau, \
    ModelCheckpoint

from mim.util.logs import get_logger
from mim.util.util import keras_model_summary_as_string
from mim.experiments.extractor import DataWrapper
from mim.util.metrics import rule_out

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
            checkpoint_path=None,
            random_state=123
    ):
        self.model = model
        self.can_use_tf_dataset = can_use_tf_dataset
        if checkpoint_path:
            self.checkpoint_path = checkpoint_path
        else:
            self.checkpoint_path = ""

        if hasattr(model, "random_state"):
            model.random_state = random_state

    def predict(self, data: DataWrapper):
        x = data.x(self.can_use_tf_dataset)
        prediction = self._prediction(x)

        if self.only_last_prediction_column_is_used:
            prediction = prediction[:, 1]

        return data.to_dataframe(prediction)

    def fit(self, training_data, validation_data=None, **kwargs):
        if self.can_use_tf_dataset:
            train = training_data.as_dataset(**kwargs)
            val = validation_data.as_dataset(**kwargs)
            return self.model.fit(train, validation_data=val, **kwargs).history
        else:
            self.model.fit(*training_data.as_numpy())
            return None

    @property
    def summary(self):
        return None

    @property
    def only_last_prediction_column_is_used(self):
        return (self.model_type is ModelTypes.CLASSIFIER and
                len(self.model.classes_) == 2)

    def save(self, split_number):
        name = 'model.sklearn'
        if split_number is None:
            split_folder = ""
        else:
            split_folder = f'split_{split_number}'

        path = os.path.join(self.checkpoint_path, split_folder)

        # If the folder doesn't exist yet, create it!
        # https://stackoverflow.com/a/273227
        Path(path).mkdir(parents=True, exist_ok=True)

        checkpoint = os.path.join(path, name)
        pd.to_pickle(self.model, checkpoint)

    def _prediction(self, x):
        return self.model.predict_proba(tf.convert_to_tensor(x))


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


class PredictionLogger(keras.callbacks.Callback):
    def __init__(self, train, val):
        super().__init__()
        self.training_data = train.x(can_use_tf_dataset=True).batch(32)
        self.validation_data = val.x(can_use_tf_dataset=True).batch(32)

    def on_epoch_end(self, epoch, logs=None):
        t0 = time()
        logs["predictions"] = _fix_prediction(
            self.model(self.training_data))
        if self.validation_data:
            logs["val_predictions"] = _fix_prediction(
                self.model(self.validation_data))
        log.info(f"PredictionCallback time: {time() - t0}")


class RuleOutLogger(keras.callbacks.Callback):
    def __init__(self, train, val):
        super().__init__()
        self.x_train = train.x(can_use_tf_dataset=True).batch(32)
        self.x_val = val.x(can_use_tf_dataset=True).batch(32)
        self.y_train = train.y
        self.y_val = val.y

    def on_epoch_end(self, epoch, logs=None):
        pred_train = _fix_prediction(
            self.model.predict(self.x_train, verbose=0))
        pred_val = _fix_prediction(
            self.model.predict(self.x_val, verbose=0))
        logs['rule_out'] = rule_out(self.y_train, pred_train)
        logs['val_rule_out'] = rule_out(self.y_val, pred_val)


def _fix_prediction(prediction):
    if isinstance(prediction, list):
        return np.concatenate(prediction, axis=1)
    else:
        return prediction


class KerasWrapper(Model):
    def __init__(
            self,
            model: tf.keras.Model,
            batch_size=16,
            epochs=2,
            initial_epoch=0,
            optimizer='adam',
            loss='binary_crossentropy',
            loss_weights=None,
            metrics=None,
            rule_out_logger=False,
            ignore_callbacks=False,
            save_prediction_history=False,
            save_model_checkpoints=False,
            use_tensorboard=False,
            save_learning_rate=False,
            checkpoint_path=None,
            skip_compile=False,
            tensorboard_path=None,
            exp_base_path=None,
            class_weight=None,
            reduce_lr_on_plateau=None,
            plot_model=True,
            **kwargs
    ):
        super().__init__(model, can_use_tf_dataset=True, **kwargs)
        if not skip_compile:
            log.debug("Compiling model!")
            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                loss_weights=loss_weights,
                metrics=metrics
            )
        else:
            log.debug("Skipping model compile!")

        self.checkpoint_path = checkpoint_path
        self.tensorboard_path = tensorboard_path
        self.exp_base_path = exp_base_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_epoch = initial_epoch
        self.ignore_callbacks = ignore_callbacks
        self.save_prediction_history = save_prediction_history
        self.save_model_checkpoints = save_model_checkpoints
        self.use_tensorboard = use_tensorboard
        self.save_learning_rate = save_learning_rate
        self.class_weight = class_weight
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.rule_out_logger = rule_out_logger
        self.plot_model = plot_model
        # log.info("\n\n" + keras_model_summary_as_string(self.model))

    def fit(self, training_data, validation_data=None, split_number=None,
            **kwargs):
        if self.plot_model:
            keras.utils.plot_model(
                self.model,
                os.path.join(self.exp_base_path, "network-graph.png"),
                True,
                True
            )
        if self.ignore_callbacks:
            callbacks = None
        else:
            callbacks = self._init_callbacks(
                split_number, training_data, validation_data
            )
        if self.batch_size < 0:
            self.batch_size = len(training_data)
        return super().fit(
            training_data,
            validation_data=validation_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            initial_epoch=self.initial_epoch,
            callbacks=callbacks,
            class_weight=self.class_weight,
            **kwargs
        )

    def _init_callbacks(self, split_number, training_data, validation_data):
        callbacks = []
        if split_number is None:
            split_folder = ""
        else:
            split_folder = f'split_{split_number}'

        if self.save_prediction_history:
            callbacks.append(
                PredictionLogger(training_data, validation_data)
            )
        if self.save_model_checkpoints:
            path = os.path.join(self.checkpoint_path, split_folder)
            callbacks.append(ModelCheckpoint(
                filepath=os.path.join(path, 'last.ckpt')))
            callbacks.append(ModelCheckpoint(
                filepath=os.path.join(path, 'last.ckpt')))
        if self.use_tensorboard:
            path = os.path.join(self.tensorboard_path, split_folder)
            callbacks.append(TensorBoard(log_dir=path))
        if self.save_learning_rate:
            callbacks.append(LearningRateLogger())
        if self.reduce_lr_on_plateau is not None:
            callbacks.append(
                ReduceLROnPlateau(**self.reduce_lr_on_plateau)
            )
        if self.rule_out_logger:
            callbacks.append(
                RuleOutLogger(training_data, validation_data)
            )
        return callbacks

    @property
    def summary(self):
        return keras_model_summary_as_string(self.model)

    @property
    def only_last_prediction_column_is_used(self):
        return False

    def save(self, split_number):
        name = "model.keras"
        if split_number is None:
            split_folder = ""
        else:
            split_folder = f'split_{split_number}'

        checkpoint = os.path.join(self.checkpoint_path, split_folder)
        self.model.save(os.path.join(checkpoint, name))

    def _prediction(self, x):
        prediction = self.model.predict(x.batch(self.batch_size))
        return _fix_prediction(prediction)
