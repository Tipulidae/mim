# -*- coding: utf-8 -*-

import os
from pathlib import Path
from enum import Enum
# from time import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard, ReduceLROnPlateau, \
    ModelCheckpoint
from sklearn.metrics import roc_auc_score

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
        self.random_state = random_state

    def predict(self, data: DataWrapper):
        x = data.x(self.can_use_tf_dataset)
        prediction = self._prediction(x)

        if self.only_last_prediction_column_is_used:
            prediction = prediction[:, 1]

        return data.to_dataframe(prediction)

    def fit(self, training_data, validation_data=None, **kwargs):
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
    def __init__(self, train, val, batch_size=32, prediction_history=None,
                 save_train=False, save_val=False, save_auc=True):
        super().__init__()
        self.training_data = train.x(can_use_tf_dataset=True).batch(batch_size)
        self.validation_data = val.x(can_use_tf_dataset=True).batch(batch_size)
        self.training_targets = train.y
        self.validation_targets = val.y
        self.prediction_history = prediction_history
        self.save_train = save_train
        self.save_val = save_val
        self.save_auc = save_auc
        if save_train:
            self.prediction_history['predictions'] = []
        if save_val:
            self.prediction_history['val_predictions'] = []

    def on_epoch_end(self, epoch, logs=None):
        # t0 = time()
        # This doesn't work anymore: the predictions can't be stored in
        # the logs dict now for some reason. Could save it manually to disk
        # instead I suppose.
        if self.save_train:
            preds = _fix_prediction(self.model.predict(
                self.training_data, verbose=0))
            self.prediction_history["predictions"].append(preds)
            if self.save_auc:
                logs['real_auc'] = roc_auc_score(self.training_targets, preds)
        if self.save_val:
            preds = _fix_prediction(self.model.predict(
                self.validation_data, verbose=0))
            self.prediction_history["val_predictions"].append(preds)
            if self.save_auc:
                logs['val_real_auc'] = roc_auc_score(
                    self.validation_targets, preds)


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


class UnfreezeModel(keras.callbacks.Callback):
    def __init__(self, unfreeze_epoch):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch != self.unfreeze_epoch:
            return

        log.debug('Unfreezing model layers')
        self.model.trainable = True
        _ = self.model.make_train_function(force=True)


class FLLogger(keras.callbacks.Callback):
    def __init__(self, train, val):
        super().__init__()
        self.x_train = train.x(can_use_tf_dataset=True)
        self.x_val = val.x(can_use_tf_dataset=True)
        self.y_train = train.y
        self.y_val = val.y

    def on_epoch_end(self, epoch, logs=None):
        pred_train = _fix_prediction(
            self.model.predict(self.x_train, verbose=0))
        pred_val = _fix_prediction(
            self.model.predict(self.x_val, verbose=0))

        logs['rule_out'] = rule_out(self.y_train, pred_train)
        logs['val_rule_out'] = rule_out(self.y_val, pred_val)


def precision_at_recall_threshold(targets, predictions, target_recall):
    tp = predictions[targets.values == 1.0]
    threshold = np.percentile(tp, 100 - target_recall*100)
    return threshold


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
            save_train_prediction_history=False,
            save_val_prediction_history=False,
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
            unfreeze_after_epoch=-1,
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
        self.save_train_prediction_history = save_train_prediction_history
        self.save_val_prediction_history = save_val_prediction_history
        self.save_model_checkpoints = save_model_checkpoints
        self.use_tensorboard = use_tensorboard
        self.save_learning_rate = save_learning_rate
        self.class_weight = class_weight
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.rule_out_logger = rule_out_logger
        self.plot_model = plot_model
        self.unfreeze_after_epoch = unfreeze_after_epoch
        self.prediction_history = {}
        self.log_real_auc = loss == 'binary_crossentropy'

    def fit(self, training_data, validation_data=None, split_number=None,
            **kwargs):
        self.prediction_history = {}
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

        train = training_data.as_dataset(
            batch_size=self.batch_size,
            **kwargs
        )
        val = validation_data.as_dataset(
            batch_size=self.batch_size,
            **kwargs
        )
        # print(f"{train=}, {val=}")
        # return self.model.fit(train, validation_data=val, **kwargs).history
        self.model.fit(
            train,
            validation_data=val,
            batch_size=self.batch_size,
            epochs=self.epochs,
            initial_epoch=self.initial_epoch,
            callbacks=callbacks,
            class_weight=self.class_weight,
            **kwargs
        )
        # This is honestly just daft. Why did they stop supporting vectors in
        # the logs?
        history = deepcopy(self.model.history.history)
        history |= self.prediction_history

        return history

    def _init_callbacks(self, split_number, training_data, validation_data):
        callbacks = []
        if split_number is None:
            split_folder = ""
        else:
            split_folder = f'split_{split_number}'

        if self.save_train_prediction_history or \
                self.save_val_prediction_history:
            callbacks.append(
                PredictionLogger(
                    training_data,
                    validation_data,
                    batch_size=self.batch_size,
                    prediction_history=self.prediction_history,
                    save_train=self.save_train_prediction_history,
                    save_val=self.save_val_prediction_history,
                    save_auc=self.log_real_auc
                )
            )
        if self.save_model_checkpoints:
            path = os.path.join(self.checkpoint_path, split_folder,
                                'checkpoints')
            if isinstance(self.save_model_checkpoints, dict):
                callbacks.append(ModelCheckpoint(
                    filepath=os.path.join(path, 'epoch_{epoch:03d}.h5'),
                    **self.save_model_checkpoints
                ))
            else:
                callbacks.append(ModelCheckpoint(
                    filepath=os.path.join(path, 'last.ckpt')))
                callbacks.append(ModelCheckpoint(
                    filepath=os.path.join(path, 'best.ckpt'),
                    save_best_only=True))
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
        if self.unfreeze_after_epoch > 0:
            callbacks.append(
                UnfreezeModel(
                    unfreeze_epoch=self.unfreeze_after_epoch,
                ))

        return callbacks

    @property
    def summary(self):
        return keras_model_summary_as_string(self.model)

    @property
    def only_last_prediction_column_is_used(self):
        return False

    def save(self, split_number):
        name = "model.tf"
        if split_number is None:
            split_folder = ""
        else:
            split_folder = f'split_{split_number}'

        checkpoint = os.path.join(self.checkpoint_path, split_folder)
        self.model.save(os.path.join(checkpoint, name), save_format='tf')

    def _prediction(self, x):
        prediction = self.model.predict(x.batch(self.batch_size))
        return _fix_prediction(prediction)


def get_torch_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


class TorchWrapper(Model):
    def __init__(
            self,
            model,
            checkpoint_path=None,
            batch_size=1,
            epochs=1,
            optimizer=None,
            optimizer_kwargs={},
            learning_rate=1e-3,
            loss=None,
            save_train_prediction_history=False,
            save_val_prediction_history=False,
            save_learning_rate=False,
            prefetch=None,
            random_state=123):
        super().__init__(model, can_use_tf_dataset=False,
                         random_state=random_state)
        device = get_torch_device()
        log.debug(f'Using pytorch on {device}')
        self.model.to(device)
        self.loss = loss
        self.optimizer = optimizer(
            model.parameters(), lr=learning_rate, **optimizer_kwargs)
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_train_prediction_history = save_train_prediction_history
        self.save_val_prediction_history = save_val_prediction_history
        self.save_learning_rate = save_learning_rate
        self.prefetch = prefetch

    def predict(self, data: DataWrapper):
        dataloader = data.as_dataloader(
            batch_size=self.batch_size,
            prefetch=self.prefetch,
            shuffle=False,
        )
        device = get_torch_device()
        predictions = []
        for _, (x, y) in enumerate(dataloader):
            x = x.to(device)
            predictions.append(self.model(x).numpy(force=True))

        # x = torch.utils.data.Dataloader(
        #     data.data['x'], batch_size=1, shuffle=False
        # )
        predictions = np.concatenate(predictions, axis=0)
        # print(predictions, predictions.shape)
        return predictions
        # x = data.x(self.can_use_tf_dataset)
        # prediction = self._prediction(x)

        # if self.only_last_prediction_column_is_used:
        #     prediction = prediction[:, 1]

        # return data.to_dataframe(prediction)

    def fit(self, training_data, validation_data=None, verbose=0,
            split_number=0):
        train = training_data.as_dataloader(
            batch_size=self.batch_size,
            prefetch=self.prefetch,
            shuffle=True,
            random_seed=self.random_state
        )
        val = validation_data.as_dataloader(
            batch_size=self.batch_size,
            prefetch=self.prefetch,
            shuffle=False,
        )
        device = get_torch_device()
        history = {'loss': [], 'val_loss': []}
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}\n---------")
            history['loss'].append(self.training_loop(train, device))
            history['val_loss'].append(self.validation_loop(val, device))

        print("Finished training pytorch model!")
        # should return the training history as a dict.
        return history

    def training_loop(self, dataloader, device):
        self.model.train()
        num_batches = len(dataloader)
        loss_sum = 0.0
        loss_avg = 0
        steps_total = 0
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            pred = self.model(x)
            train_loss = self.loss(pred, y)
            train_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_sum += train_loss.item()
            steps_total += len(x)
            loss_avg = loss_sum / steps_total

            print(f"Loss: {loss_avg:>7f} "
                  f"[{batch + 1:>5d} / {num_batches:>5d}]")

        return loss_avg

    def validation_loop(self, dataloader, device):
        self.model.eval()
        loss_sum = 0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                pred = self.model(x)
                loss_sum += self.loss(pred, y).item()

        loss_avg = loss_sum / len(dataloader.dataset)
        print(f"Validation loss: {loss_avg}")
        return loss_avg

    @property
    def summary(self):
        return str(self.model)
