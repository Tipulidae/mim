# -*- coding: utf-8 -*-

import os
import shutil
import re
import math
import random
from copy import copy
from time import time
from pathlib import Path
from glob import glob
from typing import Any, Tuple, NamedTuple, Callable, Union

import numpy as np
import pandas as pd
import torch
import silence_tensorflow.auto  # noqa: F401
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit, KFold

from mim.experiments.extractor import Extractor, Augmentor
from mim.cross_validation import CrossValidationWrapper, \
    RepeatingCrossValidator
from mim.config import PATH_TO_TEST_RESULTS
from mim.model_wrapper import Model, KerasWrapper, TorchWrapper
from mim.util.logs import get_logger
from mim.util.metadata import Metadata, Validator
from mim.util.util import callable_to_string, keras_model_summary_as_string
from mim.experiments.results import ExperimentResult, TestResult, Result
import mim.experiments.hyper_parameter as hp

log = get_logger("Experiment")


class Experiment(NamedTuple):
    description: str
    extractor: Callable[[Any], Extractor] = None
    extractor_index: dict = {}
    extractor_features: dict = {}
    extractor_labels: dict = {}
    extractor_processing: dict = {}
    data_fits_in_memory: bool = True
    augmentation: Callable[[Any], Augmentor] = None
    augmentation_kwargs: dict = {}
    use_predefined_splits: bool = False
    cv: Any = KFold
    cv_kwargs: dict = {}
    model: Any = RandomForestClassifier
    model_kwargs: dict = {}
    save_model: bool = True
    save_results: bool = True
    building_model_requires_development_data: bool = False
    optimizer: Any = 'adam'
    optimizer_kwargs: dict = {}
    learning_rate: Any = 0.01
    loss: Any = 'binary_crossentropy'
    loss_kwargs: Any = None
    loss_weights: Any = None
    class_weight: Union[dict, hp.Param] = None
    epochs: Union[int, hp.Param] = None
    initial_epoch: int = 0
    batch_size: Union[int, hp.Param] = 64
    metrics: Any = ['accuracy', 'auc']
    skip_compile: bool = False
    random_state: Union[int, hp.Param] = 123
    scoring: Any = roc_auc_score
    log_environment: bool = True
    alias: str = ''
    parent_base: str = None
    parent_name: str = None
    pre_processor: Any = None
    pre_processor_kwargs: dict = {}
    reduce_lr_on_plateau: Any = None
    ignore_callbacks: bool = False
    save_train_pred_history: bool = False
    save_val_pred_history: bool = False
    model_checkpoints: dict = None  # None means no checkpoints saved
    test_model: str = 'last'  # last, best or epoch_###
    use_tensorboard: bool = False
    save_learning_rate: bool = False
    unfreeze_after_epoch: int = -1
    ensemble: int = 1
    rule_out_logger: bool = False
    verbose: int = 1

    def run(self, action='train', restart=False, splits_to_do=-1):
        try:
            # Wipe all old results here!
            if action == 'train':
                if restart:
                    self.clear_old_results()
                    self.make_results_folder()
                elif not os.path.exists(self.base_path):
                    self.make_results_folder()

                results = self._train_and_validate(splits_to_do)
                path = self.train_result_path
            elif action == 'test':
                results = self._evaluate()
                path = self.test_result_path
            else:
                raise ValueError(
                    f'Invalid action, {action}, must be either train '
                    f'or test!'
                )

            if self.save_results:
                pd.to_pickle(results, path)
                log.debug(f'Saved results in {path}\n\n\n')
            return results
        except Exception as e:
            log.error('Something went wrong!')
            raise e

    def clear_old_results(self):
        if os.path.exists(self.base_path):
            log.debug(f'Removing old experiment results from '
                      f'{self.base_path}')
            shutil.rmtree(self.base_path, ignore_errors=True)
        else:
            log.debug('No old experiment results found.')

    def make_results_folder(self):
        os.makedirs(self.base_path, exist_ok=True)

    def _evaluate(self) -> TestResult:
        """
        Evaluate the experiment on the test set. This will load a previously
        trained model, which requires the _train_and_validate step to have
        finished. After loading the model (or models, if there are more than
        one split in the cross-validation), evaluate it on the test-set.
        """
        log.info(f'Evaluating experiment {self.name}: {self.description}')
        extractor = self.get_extractor()

        # Unfortunately, for now, I will have to load both development and
        # test sets, so that I can pre-process the test set using parameters
        # estimated from the development set. Ideally, I will move the
        # pre-processing to be part of the model-wrapper and automatically
        # save any pre-processing parameters along with the actual model,
        # but this requires more re-factoring than I have time for right now.
        if self.pre_processor is None:
            test = extractor.get_test_data()
        else:
            dev = extractor.get_development_data()
            test = extractor.get_test_data()
            pre_process = self.get_pre_processor(0)
            _, test = pre_process(dev, test)

        targets = test.y
        predictions = []
        for path in self._model_paths():
            model = self.load_model(path, target_columns=test.target_columns)
            prediction = model.predict(test)
            predictions.append(prediction)

        predictions = pd.concat(
            predictions,
            axis=1,
            keys=range(len(predictions)),
            names=['split', 'target']
        )

        results = TestResult(
            metadata=Metadata().report(conda=self.log_environment),
            targets=targets,
            predictions=predictions
        )
        return results

    def _model_paths(self):
        # test_model should be either 'last', 'best' or 'epoch_###'
        # for a specific epoch.

        # If we want the last model, use the model saved as 'model'. Might
        # want to refactor this, but would require re-running a lot of stuff
        # maybe.
        if self.test_model == 'last':
            model = 'model'
        else:
            model = self.test_model

        return glob(os.path.join(
            self.base_path, f'split*/**/{model}.*'))

    def _train_and_validate(self, splits_to_do=-1) -> ExperimentResult:
        """
        Load all the necessary data, split according to the specified
        cross-validation function. Build the model, then train using the
        training data and evaluate on the validation data.
        """
        log.info(f'Running experiment {self.name}: {self.description}')
        t = time()

        data = self.get_extractor().get_development_data()
        cv = self.get_cross_validation(data.predefined_splits)

        if self.has_train_results:
            results = pd.read_pickle(self.train_result_path)
            md = Metadata().report(conda=self.log_environment)
            Validator(
                allow_uncommitted=False,
                allow_different_commits=False,
                allow_different_branches=False,
                allow_different_environments=False
            ).validate_consistency([md, results.metadata])
        else:
            results = ExperimentResult(
                feature_names=data.feature_names,
                metadata=Metadata().report(conda=self.log_environment),
                experiment_summary=self.asdict(),
                path=self.base_path,
                total_splits=cv.get_n_splits()
            )

        splits_so_far = results.num_splits_done
        splits_total = cv.get_n_splits()
        if splits_to_do == -1:
            splits_to_do = splits_total - splits_so_far

        for i, (train, validation) in enumerate(cv.split(data)):
            if i < splits_so_far or i >= splits_to_do + splits_so_far:
                continue

            train, validation = self.augment_data(train, validation)
            pre_process = self.get_pre_processor(i)
            train, validation = pre_process(train, validation)
            model = self.build_model(train, validation, split_number=i)
            results.model_summary = model.summary
            train_result, validation_result = train_model(
                train,
                validation,
                model,
                self.scoring,
                split_number=i,
                total_splits=splits_total,
                save_model=self.save_model,
                verbose=self.verbose
            )
            results.add(train_result, validation_result)
            if self.save_results:
                pd.to_pickle(results, self.train_result_path)

        log.info(f'Finished computing scores for {self.name} in '
                 f'{time() - t}s. ')
        return results

    def get_pre_processor(self, split=0):
        if self.pre_processor is None:
            return lambda x, y: (x, y)

        return self.pre_processor(
            split_number=split, **self.pre_processor_kwargs)

    def get_extractor(self) -> Extractor:
        """
        Create and return the Extractor

        :return: Extractor object
        """
        return self.extractor(
            index=self.extractor_index,
            features=self.extractor_features,
            labels=self.extractor_labels,
            processing=self.extractor_processing,
            fits_in_memory=self.data_fits_in_memory
        )

    def augment_data(self, train, validation):
        if self.augmentation is None:
            return train, validation

        log.info('Applying data-augmentation')
        augmentor = self.augmentation(**self.augmentation_kwargs)
        train = augmentor.augment_training_data(train)
        validation = augmentor.augment_validation_data(validation)
        return train, validation

    def get_cross_validation(self, predefined_splits=None):
        if self.use_predefined_splits:
            if predefined_splits is None:
                raise ValueError(
                    "Specified use_predefined_splits, but Data has no "
                    "predefined_splits!"
                )
            cv = PredefinedSplit
            cv_kwargs = {'test_fold': predefined_splits}
        else:
            cv = self.cv
            cv_kwargs = self.cv_kwargs

        if self.ensemble > 1:
            cv_kwargs = {
                'cv': cv,
                'cv_kwargs': cv_kwargs,
                'repeats': self.ensemble
            }
            cv = RepeatingCrossValidator

        return CrossValidationWrapper(cv(**cv_kwargs))

    def build_model(self, train, validation, split_number):
        if self.has_partially_trained_model(split_number):
            resume_from_epoch, path = self.get_partial_epoch(split_number)
            model = self.load_partially_trained_model(path)
            reset_random_generators(self.random_state + split_number)
        else:
            resume_from_epoch = 0
            model_kwargs = copy(self.model_kwargs)

            if self.building_model_requires_development_data:
                model_kwargs['train'] = train
                model_kwargs['validation'] = validation

            # This is kinda ugly, but important that the model is loaded from
            # the right split, otherwise we peek!
            if self.model.__name__ == 'load_keras_model':
                model_kwargs['split_number'] = split_number

            reset_random_generators(self.random_state + split_number)
            model = self.model(**model_kwargs)

        train_size = 0 if train is None else len(train)
        return self._wrap_model(
            model, train_size=train_size, resume_from_epoch=resume_from_epoch,
            target_columns=train.target_columns
        )

    def _make_optimizer(self, train_size=0):
        if isinstance(self.learning_rate, float):
            lr = self.learning_rate
        else:
            kwargs = copy(self.learning_rate['kwargs'])
            if 'steps_per_epoch' in kwargs:
                kwargs['steps_per_epoch'] = math.ceil(
                    train_size / self.batch_size)

            lr = self.learning_rate['scheduler'](**kwargs)
        optimizer = self.optimizer(
            learning_rate=lr,
            **self.optimizer_kwargs
        )
        return optimizer

    def _wrap_model(self, model, train_size=0, target_columns=None,
                    resume_from_epoch=0, verbose=1):
        if isinstance(model, tf.keras.Model):
            optimizer = self._make_optimizer(train_size=train_size)

            if callable(self.loss):
                loss = self.loss(**self.loss_kwargs)
            else:
                loss = self.loss

            wrapped_model = KerasWrapper(
                model,
                # TODO: Add data augmentation here maybe, and use in fit
                checkpoint_path=self.base_path,
                tensorboard_path=self.base_path,
                exp_base_path=self.base_path,
                batch_size=self.batch_size,
                epochs=self.epochs,
                initial_epoch=self.initial_epoch + resume_from_epoch,
                optimizer=optimizer,
                loss=loss,
                loss_weights=self.loss_weights,
                class_weight=self.class_weight,
                metrics=fix_metrics(self.metrics),
                skip_compile=any([self.skip_compile, resume_from_epoch > 0]),
                ignore_callbacks=self.ignore_callbacks,
                save_train_prediction_history=self.save_train_pred_history,
                save_val_prediction_history=self.save_val_pred_history,
                model_checkpoints=self.model_checkpoints,
                use_tensorboard=self.use_tensorboard,
                save_learning_rate=self.save_learning_rate,
                reduce_lr_on_plateau=self.reduce_lr_on_plateau,
                rule_out_logger=self.rule_out_logger,
                unfreeze_after_epoch=self.unfreeze_after_epoch
            )

            if verbose:
                log.info("\n\n" + keras_model_summary_as_string(model))

            return wrapped_model
        elif isinstance(model, torch.nn.Module):
            wrapped_model = TorchWrapper(
                model,
                checkpoint_path=self.base_path,
                tensorboard_path=self.base_path,
                # exp_base_path=self.base_path,
                batch_size=self.batch_size,
                epochs=self.epochs,
                # initial_epoch=self.initial_epoch + resume_from_epoch,
                optimizer=self.optimizer,
                optimizer_kwargs=self.optimizer_kwargs,
                learning_rate=self.learning_rate,
                loss=self.loss,
                loss_kwargs=self.loss_kwargs,
                loss_weights=self.loss_weights,
                target_columns=target_columns,
                metrics=self.metrics,
                # class_weight=self.class_weight,
                # metrics=fix_metrics(self.metrics),
                # skip_compile=any([self.skip_compile, resume_from_epoch > 0]),
                # ignore_callbacks=self.ignore_callbacks,
                save_train_prediction_history=self.save_train_pred_history,
                save_val_prediction_history=self.save_val_pred_history,
                model_checkpoints=self.model_checkpoints,
                # use_tensorboard=self.use_tensorboard,
                save_learning_rate=self.save_learning_rate,
                # reduce_lr_on_plateau=self.reduce_lr_on_plateau,
                # rule_out_logger=self.rule_out_logger,
                unfreeze_after_epoch=self.unfreeze_after_epoch
            )
            return wrapped_model
        else:
            return Model(
                model,
                checkpoint_path=self.base_path,
                random_state=self.random_state
            )

    def load_model(self, path, target_columns):
        def _load():
            model_type = path.split('.')[-1]
            if model_type == 'sklearn':
                return pd.read_pickle(path)
            elif model_type in ['keras', 'tf']:
                return keras.models.load_model(filepath=path)
            elif model_type == 'pt':
                return torch.load(path)
            raise TypeError(f'Unexpected model type {model_type}')

        return self._wrap_model(
            _load(), target_columns=target_columns, verbose=0)

    def asdict(self):
        return callable_to_string(self._asdict())

    def has_partially_trained_model(self, split_number):
        checkpoint_path = os.path.join(
            self.base_path,
            f'split_{split_number}',
            'checkpoints',
            'epoch_*.h5'
        )
        cps = list(glob(checkpoint_path))
        return 0 < len(cps) < self.epochs

    def get_partial_epoch(self, split_number):
        checkpoint_path = os.path.join(
            self.base_path,
            f'split_{split_number}',
            'checkpoints',
            'epoch_*.h5'
        )
        cps = list(glob(checkpoint_path))
        p = re.compile('epoch_0*(\\d+).h5')

        def path_to_epoch(path):
            return int(p.findall(path)[0])

        latest_path = max(cps, key=path_to_epoch)
        return path_to_epoch(latest_path), latest_path

    def load_partially_trained_model(self, path):
        return keras.models.load_model(filepath=path)

    @property
    def is_binary(self):
        return self.extractor_kwargs["labels"]['is_binary']

    @property
    def name(self):
        if hasattr(self, '_name_'):
            return self._name_
        else:
            return self.alias

    @property
    def train_result_path(self):
        return os.path.join(self.base_path, 'train_val_results.pickle')

    @property
    def test_result_path(self):
        return os.path.join(self.base_path, 'test_results.pickle')

    @property
    def base_path(self):
        parent_base = self.parent_base or self.project_name
        parent_name = self.parent_name or str(self.__class__.__name__)
        return os.path.join(
            PATH_TO_TEST_RESULTS,
            parent_base,
            parent_name,
            self.name
        )

    @property
    def project_name(self):
        # Bit of a hack. self.__class__.__module__ is something like
        # projects.transfer.experiments.
        return self.__class__.__module__.split('.')[1]

    @property
    def is_trained(self):
        return self.has_train_results and not self.is_partial

    @property
    def has_train_results(self):
        file = Path(self.train_result_path)
        return file.is_file()

    @property
    def is_evaluated(self):
        file = Path(self.test_result_path)
        return file.is_file()

    @property
    def is_partial(self):
        # either there are train results where not all splits are done
        # OR there is a checkpoints folder with fewer than epochs checkpoints
        if self.has_partially_trained_model(split_number=0):
            return True
        if self.has_train_results:
            results = pd.read_pickle(self.train_result_path)
            if results.num_splits_done < results.total_splits:
                return True
        return False

    @property
    def num_splits_done(self):
        if not self.has_train_results:
            return 0

        # trunk-ignore(bandit/B301)
        results = pd.read_pickle(self.train_result_path)
        return results.num_splits_done

    @property
    def validation_scores(self):
        if self.has_train_results:
            # trunk-ignore(bandit/B301)
            results = pd.read_pickle(self.train_result_path)
            return results.validation_scores
        else:
            return None


def reset_random_generators(new_seed):
    # Releases keras global state. Ref:
    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
    tf.keras.backend.clear_session()
    np.random.seed(new_seed)
    tf.random.set_seed(new_seed)
    random.seed(new_seed)
    torch.manual_seed(new_seed)


def _history_to_dataframe(history, data):
    # history is a list of array-likes (one for each epoch), and I want to
    # combine them into one big dataframe.
    return pd.concat(
        map(data.to_dataframe, history),
        axis=1,
        keys=range(len(history)),
        names=['epoch', 'target']
    )


def gather_results(model, history_dict, data) -> Result:
    result = Result()
    result.targets = data.y
    result.predictions = model.predict(data)
    if not history_dict:
        return result

    if 'predictions' in history_dict:
        predictions = history_dict.pop('predictions')
        result.prediction_history = _history_to_dataframe(predictions, data)

    result.history = history_dict
    return result


def _split_history(history_dict):
    training_history = {}
    validation_history = {}

    if history_dict:
        for key, value in history_dict.items():
            if 'val_' in key:
                if key == 'val_predictions':
                    validation_history['predictions'] = value
                else:
                    validation_history[key] = value
            else:
                training_history[key] = value

    return training_history, validation_history


def train_model(training_data, validation_data, model, scoring,
                split_number=None, total_splits=None, save_model=True,
                verbose=1,
                ) -> Tuple[Result, Result]:
    t0 = time()
    log.info(f'\n\nFitting classifier, split {split_number} of {total_splits}'
             f'\nTrain size: {len(training_data)}, '
             f'val size: {len(validation_data)}')

    history = model.fit(
        training_data,
        validation_data=validation_data,
        split_number=split_number,
        verbose=verbose,
    )
    train_history, val_history = _split_history(history)
    train_result = gather_results(model, train_history, training_data)
    train_result.time = time() - t0

    validation_result = gather_results(model, val_history, validation_data)
    validation_result.time = time() - train_result.time - t0

    if scoring:
        train_result.score = scoring(
            train_result.targets, train_result.predictions)
        validation_result.score = scoring(
            validation_result.targets, validation_result.predictions)
        log.debug(f'train score: {train_result.score}, '
                  f'validation score: {validation_result.score}')

    if save_model:
        model.save(split_number=split_number)

    return train_result, validation_result


def fix_metrics(m):
    if isinstance(m, list):
        return [fix_metrics(x) for x in m]
    elif isinstance(m, dict):
        return {k: fix_metrics(v) for k, v in m.items()}
    else:
        return _map_metric_string_to_object(m)


def _map_metric_string_to_object(name):
    name = name.lower()
    if name == 'auc':
        return tf.keras.metrics.AUC()
    else:
        return name
