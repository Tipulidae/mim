# -*- coding: utf-8 -*-

import os
from pathlib import Path
from enum import Enum
from time import time
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard, ReduceLROnPlateau, \
    ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, \
    mean_absolute_error

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
            model_checkpoints=None,
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
        if not isinstance(model_checkpoints, (dict, type(None))):
            raise TypeError(f"model_checkpoints must be either dict or None, "
                            f"was {type(model_checkpoints)}.")
        self.model_checkpoints = model_checkpoints
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
        if self.model_checkpoints is not None:
            path = os.path.join(
                self.checkpoint_path,
                split_folder,
                'checkpoints'
            )
            # For legacy reasons, I save as .keras if we save each epoch,
            # and as .tf when we only save the best model. This is because of
            # a bug in tensorflow where save_best_only is not compatible with
            # the keras format. Should probably try to refactor this later.
            if ('save_best_only' in self.model_checkpoints and
                    self.model_checkpoints['save_best_only']):
                path = os.path.join(path, 'best.tf')
                format = 'tf'
            else:
                path = os.path.join(path, 'epoch_{epoch:03d}.keras')
                format = 'keras'

            callbacks.append(ModelCheckpoint(
                filepath=path, save_format=format, **self.model_checkpoints))

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
        name = "model.keras"
        if split_number is None:
            split_folder = ""
        else:
            split_folder = f'split_{split_number}'

        checkpoint = os.path.join(self.checkpoint_path, split_folder)
        self.model.save(os.path.join(checkpoint, name), save_format='keras')

    def _prediction(self, x):
        prediction = self.model.predict(x.batch(self.batch_size))
        return _fix_prediction(prediction)


def get_torch_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


class History:
    def __init__(self, tensorboard_path, split_number=0, train_targets=None,
                 val_targets=None, save_train_prediction_history=False,
                 save_val_prediction_history=True, metrics=None,
                 save_learning_rate=False):
        split_folder = '' if split_number is None else f'split_{split_number}'
        self.split_number = split_number
        self.train_writer = SummaryWriter(
            log_dir=os.path.join(tensorboard_path, split_folder, 'train')
        )
        self.val_writer = SummaryWriter(
            log_dir=os.path.join(tensorboard_path, split_folder, 'validation')
        )
        self.history = defaultdict(list)
        self.train_targets = train_targets
        self.val_targets = val_targets
        self.save_train_prediction_history = save_train_prediction_history
        self.save_val_prediction_history = save_val_prediction_history
        self.save_learning_rate = save_learning_rate

        available_metrics = {
            'auc': roc_auc_score,
            'accuracy': accuracy_score,
            'r2': r2_score,
            'mae': mean_absolute_error
        }
        if isinstance(metrics, dict):
            self.metrics = {}
            for target, metric_list in metrics.items():
                self.metrics[target] = {
                    metric: available_metrics[metric] for metric in metric_list
                }
        else:
            self.metrics = {}
            for target in train_targets.columns:
                self.metrics[target] = {
                    metric: available_metrics[metric] for metric in metrics
                }

    def _log_metric(self, metrics, targets, predictions, epoch,
                    target_name=None, validation=False):
        log_strings = []
        writer = self.val_writer if validation else self.train_writer
        for metric_name, func in metrics.items():
            metric = func(targets, predictions.ravel())
            if target_name is None:
                history_label = metric_name
            else:
                history_label = f'{target_name}_{metric_name}'

            writer.add_scalar(
                f"epoch_{history_label}", metric, global_step=epoch)

            if validation:
                history_label = "val_" + history_label

            self.history[history_label].append(metric)
            log_strings.append(f"{history_label}: {metric:.4f}")

        return log_strings

    def log(self, train_loss, val_loss, train_predictions, val_predictions,
            learning_rate, epoch):
        log_strings = []
        self.history['loss'].append(train_loss)
        self.train_writer.add_scalar(
            'epoch_loss', train_loss, global_step=epoch)
        log_strings.append(f"loss: {train_loss:.4f}")

        for i, target_name in enumerate(self.train_targets.columns):
            log_strings.extend(
                self._log_metric(
                    self.metrics[target_name],
                    self.train_targets[target_name],
                    train_predictions[:, i],
                    epoch,
                    target_name=target_name
                )
            )

        self.history['val_loss'].append(val_loss)
        self.val_writer.add_scalar('epoch_loss', val_loss, global_step=epoch)
        log_strings.append(f"val_loss: {val_loss:.4f}")

        for i, target_name in enumerate(self.val_targets.columns):
            log_strings.extend(
                self._log_metric(
                    self.metrics[target_name],
                    self.val_targets[target_name],
                    val_predictions[:, i],
                    epoch,
                    target_name=target_name,
                    validation=True
                )
            )

        if self.save_learning_rate:
            self.history['lr'].append(learning_rate)
            self.train_writer.add_scalar('epoch_lr', learning_rate,
                                         global_step=epoch)
            log_strings.append(f'lr: {learning_rate:.4E}')

        if self.save_train_prediction_history:
            self.history['predictions'].append(train_predictions)
        if self.save_val_prediction_history:
            self.history['val_predictions'].append(val_predictions)

        return " - ".join(log_strings)

    def flush(self):
        self.train_writer.flush()
        self.val_writer.flush()

    def __del__(self):
        self.val_writer.close()
        self.train_writer.close()


class NullScheduler:
    def __init__(self, lr):
        self.lr = lr

    def step(self):
        pass

    def get_last_lr(self):
        return [self.lr]


class MultiLoss:
    def __init__(self, loss_dict, target_columns, loss_kwargs=None,
                 loss_weights=None):
        if loss_kwargs is None:
            loss_kwargs = {
                loss_name: {'reduction': 'sum'} for loss_name in loss_dict
            }
        if loss_weights is None:
            loss_weights = {loss_name: 1.0 for loss_name in loss_dict}

        if loss_dict.keys() != loss_kwargs.keys() != loss_weights.keys():
            raise ValueError("The loss_dict, loss_kwargs and loss_weights "
                             "dicts must contain the same set of keys.")
        if target_columns.keys() != loss_dict.keys():
            raise ValueError("The loss_dict must contain the same keys as "
                             "the targets.")

        self.loss_dict = {
            loss_name: loss_fn(**loss_kwargs[loss_name])
            for loss_name, loss_fn in loss_dict.items()
        }
        self.loss_weights = loss_weights
        self.target_order = target_columns

    def __call__(self, predictions, targets):
        losses = []
        for i, name in enumerate(self.target_order):
            losses.append(
                self.loss_dict[name](predictions[:, i], targets[:, i]) *
                self.loss_weights[name]
            )
        return sum(losses)


def init_optimizer_and_scheduler(
        model_parameters, optimizer_fn, optimizer_kwargs, learning_rate):

    if isinstance(learning_rate, float):
        optimizer = optimizer_fn(
            model_parameters, lr=learning_rate, **optimizer_kwargs)
        scheduler = NullScheduler(learning_rate)

    elif learning_rate is None:
        if 'lr' not in optimizer_kwargs:
            raise ValueError(
                "Must specify learning rate, either as 'lr' in "
                "optimizer_kwargs or by setting learning_rate to a "
                "float value."
            )
        optimizer = optimizer_fn(model_parameters, **optimizer_kwargs)
        scheduler = NullScheduler(optimizer_kwargs['lr'])
    else:
        if 'scheduler' not in learning_rate:
            raise ValueError(
                "When supplying non-float learning-rate to pytorch "
                "models, you need to include a scheduler as a keyword "
                "argument to learning_rate."
            )
        if 'kwargs' not in learning_rate:
            raise ValueError(
                "When supplying non-float learning-rate to pytorch "
                "models, you need to include a kwargs dict for the "
                "scheduler in learning_rate."
            )
        if 'lr' in optimizer_kwargs:
            raise ValueError(
                "Don't set the lr parameter manually when using a scheduler."
            )

        optimizer = optimizer_fn(model_parameters, lr=1.0, **optimizer_kwargs)
        scheduler = learning_rate['scheduler'](
            optimizer,
            **learning_rate['kwargs']
        )

    return optimizer, scheduler


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
            loss_kwargs=None,
            loss_weights=None,
            target_columns=None,
            save_train_prediction_history=False,
            save_val_prediction_history=False,
            save_learning_rate=False,
            model_checkpoints=None,
            unfreeze_after_epoch=-1,
            prefetch=None,
            tensorboard_path=None,
            metrics=None,
            random_state=123):
        super().__init__(model, can_use_tf_dataset=False,
                         checkpoint_path=checkpoint_path,
                         random_state=random_state)
        device = get_torch_device()
        log.debug(f'Using pytorch on {device}')
        self.model.to(device)
        if isinstance(loss, dict):
            self.loss = MultiLoss(
                loss, target_columns,
                loss_kwargs=loss_kwargs,
                loss_weights=loss_weights,
            )
        elif loss_kwargs is None:
            self.loss = loss(reduction='sum')
        else:
            self.loss = loss(**loss_kwargs)

        self.optimizer, self.scheduler = init_optimizer_and_scheduler(
            model.parameters(), optimizer, optimizer_kwargs, learning_rate
        )

        self.epochs = epochs
        self.batch_size = batch_size
        self.save_train_prediction_history = save_train_prediction_history
        self.save_val_prediction_history = save_val_prediction_history
        self.save_learning_rate = save_learning_rate
        self.prefetch = prefetch
        self.tensorboard_path = tensorboard_path
        self.metrics = metrics
        self.unfreeze_after_epoch = unfreeze_after_epoch
        if model_checkpoints is None:
            self.model_checkpoint = TorchModelCheckpoint(
                checkpoint_path, save_freq='never')
        else:
            self.model_checkpoint = TorchModelCheckpoint(
                checkpoint_path, **model_checkpoints)

    def predict(self, data: DataWrapper):
        dataloader = data.as_dataloader(
            batch_size=self.batch_size,
            prefetch=self.prefetch,
            shuffle=False,
        )
        device = get_torch_device()
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for _, (x, _, _) in enumerate(dataloader):
                x = to_device(x, device)
                with torch.autocast(device_type=device):
                    predictions.append(self.model(x).numpy(force=True))

        predictions = np.concatenate(predictions, axis=0)
        return data.to_dataframe(predictions)

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
        history = History(
            self.tensorboard_path,
            split_number=split_number,
            train_targets=training_data.y,
            val_targets=validation_data.y,
            save_train_prediction_history=self.save_train_prediction_history,
            save_val_prediction_history=self.save_val_prediction_history,
            save_learning_rate=self.save_learning_rate,
            metrics=self.metrics
        )
        for epoch in range(self.epochs):
            t0 = time()
            print(f"Epoch {epoch + 1}/{self.epochs}")
            lr = self.scheduler.get_last_lr()[0]
            self.unfreeze_layers_maybe(epoch)
            train_loss, train_preds = self.training_loop(train, device)
            val_loss, val_preds = self.validation_loop(val, device)

            log_string = history.log(
                train_loss=train_loss,
                val_loss=val_loss,
                train_predictions=train_preds,
                val_predictions=val_preds,
                learning_rate=lr,
                epoch=epoch,
            )

            elapsed_time = time() - t0
            print(
                f"\r{len(train)}/{len(train):>} "
                f"[==============================]"
                f" - {elapsed_time:.1f}s - {log_string}"
            )
            self.model_checkpoint.save(self.model, epoch, history)

        history.flush()
        print("Finished training pytorch model!")
        return history.history

    def training_loop(self, dataloader, device):
        self.model.train()
        num_batches = len(dataloader)
        loss_sum = 0.0
        loss_avg = 0
        steps_total = 0
        predictions = []
        indices = []
        for batch, (x, y, idx) in enumerate(dataloader):
            t0 = time()
            x = to_device(x, device)
            y = to_device(y, device, force_float=True)
            with torch.autocast(device_type=device):
                pred = self.model(x)
                train_loss = self.loss(pred, y)

            train_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_sum += train_loss.item()
            steps_total += len(idx)
            loss_avg = loss_sum / steps_total

            progress = progress_string(batch, num_batches)
            elapsed_time = time() - t0
            print(f"\r{progress} - {elapsed_time:.1e}s/batch - "
                  f"loss: {loss_avg:.4f}",
                  end='')

            predictions.append(pred.detach().cpu().numpy())
            indices.append(idx.numpy())

        # Updates the learning-rate according to our scheduler
        self.scheduler.step()
        # The indices are collected in order to undo the shuffling so we can
        # use the predictions to calculate (global) metrics like AUC.
        indices = np.concatenate(indices).astype(int)
        predictions = np.concatenate(predictions)[indices.argsort()]
        return loss_avg, predictions

    def validation_loop(self, dataloader, device):
        self.model.eval()
        loss_sum = 0
        predictions = []
        with torch.no_grad():
            for x, y, _ in dataloader:
                x = to_device(x, device)
                y = to_device(y, device, force_float=True)
                with torch.autocast(device_type=device):
                    pred = self.model(x)
                    loss_sum += self.loss(pred, y).item()

                predictions.append(pred.detach().cpu().numpy())

        loss_avg = loss_sum / len(dataloader.dataset)
        return loss_avg, np.concatenate(predictions)

    def unfreeze_layers_maybe(self, epoch):
        # This happens before the training loop for epoch starts. Epoch is
        # zero indexed, but if I say "unfreeze after epoch 4, I mean 4 as in
        # not zero-indexed. So train 4 epochs frozen, then unfreeze for epochs
        # 5 and above.
        if epoch == self.unfreeze_after_epoch:
            log.debug('Unfreezing model parameters')
            for param in self.model.parameters():
                param.requires_grad = True

    def save(self, split_number):
        name = "model.pt"
        if split_number is None:
            split_folder = ""
        else:
            split_folder = f'split_{split_number}'

        path = os.path.join(self.checkpoint_path, split_folder, name)
        torch.save(self.model, path)
        log.debug(f"Saved model to {path}")

    @property
    def summary(self):
        return str(self.model)


class TorchModelCheckpoint:
    # Trying to stick to the keras API here with the argument names
    def __init__(self, filepath, monitor='val_loss', save_best_only=False,
                 mode='min', save_freq='epoch'):

        if mode not in ['min', 'max']:
            raise ValueError(f"mode should be either min or max, was {mode}")
        if mode == 'min':
            self.op = np.less
            self.best = np.inf
        else:
            self.op = np.greater
            self.best = -np.inf

        self.save_best_only = save_best_only
        self.monitor = monitor
        self.filepath = filepath
        self.save_freq = save_freq

    def save(self, model, epoch, history):
        # I suppose I want to either save every epoch or save best epoch,
        # but not both at the same time, right?
        if not self._should_save(history):
            return
        path = os.path.join(
            self.filepath, f"split_{history.split_number}", "checkpoints")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        if self.save_best_only:
            path = os.path.join(path, 'best.pt')
        else:
            path = os.path.join(path, f'epoch_{epoch+1:03d}.pt')

        torch.save(model, path)
        log.debug('Model checkpoint saved')

    def _should_save(self, history):
        if self.save_freq == 'never':
            return False
        if not self.save_best_only:
            return True
        current = history.history[self.monitor][-1]
        if self.op(current, self.best):
            self.best = current
            return True
        else:
            return False


def to_device(x, device, force_float=False):
    if isinstance(x, dict):
        return {
            k: to_device(v, device, force_float=force_float)
            for k, v in x.items()
        }
    else:
        if force_float:
            return x.float().to(device)
        else:
            return x.to(device)


def progress_string(current_batch, total_batches):
    batch_offset = len(str(total_batches))
    bar = progress_bar(current_batch, total_batches)
    return f"{current_batch:>{batch_offset}}/{total_batches} {bar}"


def progress_bar(current_batch, total_batches):
    bar_length = 30
    done = round((bar_length - 1) * current_batch / total_batches)
    remaining = bar_length - 1 - done
    return "[" + "="*done + ">" + "."*remaining + "]"
