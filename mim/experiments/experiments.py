# -*- coding: utf-8 -*-

import os
import shutil
from copy import copy
from time import time
from pathlib import Path
from glob import glob
from typing import Any, Tuple, NamedTuple, Callable, Union

import numpy as np
import pandas as pd
import silence_tensorflow.auto  # noqa: F401
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit, KFold

from mim.experiments.extractor import Extractor
from mim.cross_validation import CrossValidationWrapper, \
    RepeatingCrossValidator
from mim.config import PATH_TO_TEST_RESULTS
from mim.model_wrapper import Model, KerasWrapper
from mim.util.logs import get_logger
from mim.util.metadata import Metadata, Validator
from mim.util.util import callable_to_string
from mim.experiments.results import ExperimentResult, TestResult, Result
import mim.experiments.hyper_parameter as hp

log = get_logger("Experiment")


class Experiment(NamedTuple):
    description: str
    extractor: Callable[[Any], Extractor] = None
    extractor_kwargs: dict = {
        "index": {},
        "features": None,
        "labels": None,
        "processing": None,
    }
    use_predefined_splits: bool = False
    cv: Any = KFold
    cv_kwargs: dict = {}
    model: Any = RandomForestClassifier
    model_kwargs: dict = {}
    save_model: bool = True
    save_results: bool = True
    building_model_requires_development_data: bool = False
    optimizer: Any = 'adam'
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
    log_conda_env: bool = True
    alias: str = ''
    parent_base: str = None
    parent_name: str = None
    data_fits_in_memory: bool = True
    pre_processor: Any = None
    pre_processor_kwargs: dict = {}
    reduce_lr_on_plateau: Any = None
    ignore_callbacks: bool = False
    save_prediction_history: bool = False
    save_model_checkpoints: bool = False
    use_tensorboard: bool = False
    save_learning_rate: bool = False
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
        dev = extractor.get_development_data()
        test = extractor.get_test_data()
        pre_process = self.get_pre_processor(0)
        dev, test = pre_process(dev, test)

        targets = test.y
        predictions = []
        for path in self._model_paths():
            model = self.load_model(path)
            prediction = model.predict(test)
            predictions.append(prediction)

        predictions = pd.concat(
            predictions,
            axis=1,
            keys=range(len(predictions)),
            names=['split', 'target']
        )

        results = TestResult(
            metadata=Metadata().report(conda=self.log_conda_env),
            targets=targets,
            predictions=predictions
        )
        return results

    def _model_paths(self):
        return glob(os.path.join(self.base_path, 'split*/model.*'))

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
            md = Metadata().report(conda=self.log_conda_env)
            Validator(
                allow_uncommitted=False,
                allow_different_commits=False,
                allow_different_branches=False,
                allow_different_environments=False
            ).validate_consistency([md, results.metadata])
        else:
            results = ExperimentResult(
                feature_names=data.feature_names,
                metadata=Metadata().report(conda=self.log_conda_env),
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
        return self.extractor(**self.extractor_kwargs)

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
        model_kwargs = copy(self.model_kwargs)

        if self.building_model_requires_development_data:
            model_kwargs['train'] = train
            model_kwargs['validation'] = validation

        # This is kinda ugly, but important that the model is loaded from
        # the right split, otherwise we peek!
        if self.model.__name__ == 'load_keras_model':
            model_kwargs['split_number'] = split_number

        # Releases keras global state. Ref:
        # https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
        tf.keras.backend.clear_session()
        rand_state = self.random_state + split_number
        np.random.seed(rand_state)
        tf.random.set_seed(rand_state)
        model = self.model(**model_kwargs)
        return self._wrap_model(model)

    def _wrap_model(self, model):
        if isinstance(model, tf.keras.Model):
            # TODO: refactor this! :(
            if isinstance(self.optimizer, dict):
                optimizer_kwargs = copy(self.optimizer['kwargs'])
                optimizer = self.optimizer['name']
                if 'learning_rate' in optimizer_kwargs:
                    lr = optimizer_kwargs.pop('learning_rate')
                    if isinstance(lr, dict):
                        lr = lr['scheduler'](**lr['scheduler_kwargs'])
                    optimizer_kwargs['learning_rate'] = lr
                optimizer = optimizer(**optimizer_kwargs)
            else:
                optimizer = self.optimizer

            if callable(self.loss):
                loss = self.loss(**self.loss_kwargs)
            else:
                loss = self.loss

            return KerasWrapper(
                model,
                # TODO: Add data augmentation here maybe, and use in fit
                checkpoint_path=self.base_path,
                tensorboard_path=self.base_path,
                exp_base_path=self.base_path,
                batch_size=self.batch_size,
                epochs=self.epochs,
                initial_epoch=self.initial_epoch,
                optimizer=optimizer,
                loss=loss,
                loss_weights=self.loss_weights,
                class_weight=self.class_weight,
                metrics=fix_metrics(self.metrics),
                skip_compile=self.skip_compile,
                ignore_callbacks=self.ignore_callbacks,
                save_prediction_history=self.save_prediction_history,
                save_model_checkpoints=self.save_model_checkpoints,
                use_tensorboard=self.use_tensorboard,
                save_learning_rate=self.save_learning_rate,
                reduce_lr_on_plateau=self.reduce_lr_on_plateau,
                rule_out_logger=self.rule_out_logger
            )
        else:
            return Model(
                model,
                checkpoint_path=self.base_path,
                random_state=self.random_state
            )

    def load_model(self, path):
        def _load():
            model_type = path.split('.')[-1]
            if model_type == 'sklearn':
                return pd.read_pickle(path)
            elif model_type == 'keras':
                return keras.models.load_model(filepath=path)
            raise TypeError(f'Unexpected model type {model_type}')

        return self._wrap_model(_load())

    def asdict(self):
        return callable_to_string(self._asdict())

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
        parent_base = self.parent_base or ''
        parent_name = self.parent_name or str(self.__class__.__name__)
        return os.path.join(
            PATH_TO_TEST_RESULTS,
            parent_base,
            parent_name,
            self.name
        )

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
        if not self.has_train_results:
            return False

        results = pd.read_pickle(self.train_result_path)
        return results.num_splits_done < results.total_splits

    @property
    def num_splits_done(self):
        if not self.has_train_results:
            return 0

        results = pd.read_pickle(self.train_result_path)
        return results.num_splits_done

    @property
    def validation_scores(self):
        if self.has_train_results:
            results = pd.read_pickle(self.train_result_path)
            return results.validation_scores
        else:
            return None


def _history_to_dataframe(history, data):
    # history is a list of dataframes, and I want to combine them into
    # one big dataframe
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
    log.info(f'\n\nFitting classifier, split {split_number} of {total_splits}')

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
