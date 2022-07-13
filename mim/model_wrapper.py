# -*- coding: utf-8 -*-

import os
from enum import Enum
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau

from mim.util.logs import get_logger
from mim.util.util import keras_model_summary_as_string
from mim.extractors.extractor import DataWrapper

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

        checkpoint = os.path.join(self.checkpoint_path, split_folder, name)
        pd.to_pickle(self.model, checkpoint)

    def _prediction(self, x):
        return self.model.predict_proba(x)


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
        logs["training_predictions"] = self.model.predict(
            self.training_data)
        if self.validation_data:
            logs["validation_predictions"] = self.model.predict(
                self.validation_data)
        log.info(f"PredictionCallback time: {time() - t0}")


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
            ignore_callbacks=False,
            checkpoint_path=None,
            skip_compile=False,
            tensorboard_path=None,
            exp_base_path=None,
            class_weight=None,
            reduce_lr_on_plateau=None,
            **kwargs
    ):
        super().__init__(model, can_use_tf_dataset=True, **kwargs)
        if not skip_compile:
            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                loss_weights=loss_weights,
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

    def fit(self, training_data, validation_data=None, split_number=None,
            **kwargs):
        keras.utils.plot_model(
            self.model,
            os.path.join(self.exp_base_path, "network-graph.png"),
            True,
            True
        )
        if self.ignore_callbacks:
            callbacks = None
        else:
            if split_number is None:
                split_folder = ""
            else:
                split_folder = f'split_{split_number}'

            # checkpoint = os.path.join(self.checkpoint_path, split_folder)
            tensorboard = os.path.join(self.tensorboard_path, split_folder)

            callbacks = [
                # ModelCheckpoint(
                #     filepath=os.path.join(checkpoint, 'last.ckpt')
                # ),
                # ModelCheckpoint(
                #     filepath=os.path.join(checkpoint, 'best.ckpt'),
                #     save_best_only=True
                # ),
                PredictionLogger(training_data, validation_data),
                TensorBoard(log_dir=tensorboard),
                LearningRateLogger()
            ]

            if self.reduce_lr_on_plateau is not None:
                callbacks.append(
                    ReduceLROnPlateau(**self.reduce_lr_on_plateau)
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
        if isinstance(prediction, list):
            return np.concatenate(prediction, axis=1)

        return prediction
