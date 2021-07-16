import os
import shutil
from copy import copy
from time import time
from pathlib import Path
from typing import Any, NamedTuple, Callable, Union

import numpy as np
import pandas as pd
import silence_tensorflow.auto  # noqa: F401
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit, KFold

from mim.extractors.extractor import Extractor, Container
from mim.cross_validation import CrossValidationWrapper
from mim.config import PATH_TO_TEST_RESULTS
from mim.model_wrapper import Model, KerasWrapper
from mim.util.logs import get_logger
from mim.util.metadata import Metadata
from mim.util.util import callable_to_string
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
    building_model_requires_development_data: bool = False
    optimizer: Any = 'adam'
    loss: Any = 'binary_crossentropy'
    class_weight: Union[dict, hp.Param] = None
    epochs: Union[int, hp.Param] = None
    initial_epoch: int = 0
    batch_size: Union[int, hp.Param] = 64
    metrics: Any = ['accuracy', 'auc']
    ignore_callbacks: bool = False
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

    def run(self):
        try:
            # Wipe all old results here!
            if os.path.exists(self.base_path):
                log.debug(f'Removing old experiment results from '
                          f'{self.base_path}')
                shutil.rmtree(self.base_path, ignore_errors=True)
            else:
                log.debug('No old experiment results found.')

            results = self._run()
            os.makedirs(self.base_path, exist_ok=True)
            pd.to_pickle(results, self.result_path)
            log.debug(f'Saved results in {self.result_path}')
        except Exception as e:
            log.error('Something went wrong!')
            raise e

    def _run(self):
        log.info(f'Running experiment {self.name}: {self.description}')
        t = time()

        data = self.get_data()

        feature_names = data.columns
        results = {
            'fit_time': [],
            'score_time': [],
            'train_score': [],
            'test_score': [],
            'targets': [],
            'predictions': [],
            'feature_names': feature_names,
            'feature_importance': [],
            'model_summary': [],
            'history': [],
        }

        cv = self.get_cross_validation(data.predefined_splits)

        for i, (train, validation) in enumerate(cv.split(data)):
            pre_process = self.get_pre_processor(i)
            train = pre_process(train)
            validation = pre_process(validation)
            result = _validate(
                train,
                validation,
                self.get_model(train, validation, split_number=i),
                self.scoring,
                split_number=i
            )
            _update_results(results, result)

        _finish_results(results)
        results['metadata'] = Metadata().report(conda=self.log_conda_env)
        results['experiment_summary'] = self.asdict()
        log.info(f'Finished computing scores for {self.name} in '
                 f'{time() - t}s. ')
        return results

    def get_pre_processor(self, split=0):
        if self.pre_processor is None:
            return lambda x: x

        return self.pre_processor(
            split_number=split, **self.pre_processor_kwargs)

    def get_data(self) -> Container:
        """
        Uses the extractor and specifications to create the X and y data
        set.

        :return: Container object
        """
        return self.extractor(**self.extractor_kwargs).get_data()

    def get_cross_validation(self, predefined_splits=None):
        if not self.use_predefined_splits and self.cv is None:
            raise ValueError("Must specify cv or use predefined splits!")

        elif self.use_predefined_splits or self.cv is None:
            if predefined_splits is None:
                raise ValueError(
                    "Data must contain predefined_splits when using "
                    "PredefinedSplit cross-validation."
                )
            cv = PredefinedSplit(predefined_splits)
        else:
            cv = self.cv(**self.cv_kwargs)

        return CrossValidationWrapper(cv)

    def get_model(self, train, validation, split_number):
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

            metric_list = []
            for metric in self.metrics:
                if metric == 'auc':
                    metric_list.append(tf.keras.metrics.AUC())
                else:
                    metric_list.append(metric)

            return KerasWrapper(
                model,
                # TODO: Add data augmentation here maybe, and use in fit
                checkpoint_path=self.base_path,
                tensorboard_path=self.base_path,
                batch_size=self.batch_size,
                epochs=self.epochs,
                initial_epoch=self.initial_epoch,
                optimizer=optimizer,
                loss=self.loss,
                class_weight=self.class_weight,
                metrics=metric_list,
                skip_compile=self.skip_compile,
                ignore_callbacks=self.ignore_callbacks,
                reduce_lr_on_plateau=self.reduce_lr_on_plateau,
            )
        else:
            model.random_state = self.random_state
            return Model(model)

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
    def result_path(self):
        return os.path.join(
            self.base_path,
            'results.pickle'
        )

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
    def is_done(self):
        file = Path(self.result_path)
        return file.is_file()

    @property
    def validation_scores(self):
        if self.is_done:
            results = pd.read_pickle(self.result_path)
            return results['test_score']
        else:
            return None


def _validate(train, val, model, scoring, split_number=None, pre_process=None):
    t0 = time()
    log.info(f'\n\nFitting classifier, split {split_number}')

    history = model.fit(train, validation_data=val, split_number=split_number)
    fit_time = time() - t0

    prediction = model.predict(val['x'])
    score_time = time() - fit_time - t0

    train_score = scoring(
        train['y'].as_numpy,
        model.predict(train['x'])['prediction']
    )

    y_val = val['y'].as_numpy
    targets = pd.DataFrame(
        y_val,
        index=pd.Index(val['index'].as_numpy, name=val['index'].columns[0]),
        columns=val['y'].columns
    )
    test_score = scoring(y_val, prediction['prediction'])
    log.debug(f'test score: {test_score}, train score: {train_score}')

    try:
        feature_importance = model.model.feature_importances_
    except AttributeError:
        feature_importance = None

    return {
        'predictions': prediction,
        'train_score': train_score,
        'test_score': test_score,
        'feature_importance': feature_importance,
        'fit_time': fit_time,
        'score_time': score_time,
        'targets': targets,
        'history': history,
        'model_summary': model.summary
    }


def _update_results(results, result):
    for key, value in result.items():
        results[key].append(value)


def _finish_results(results):
    predictions = results['predictions']
    new_predictions = {}
    for prediction in predictions:
        for key, value in prediction.items():
            if key not in new_predictions:
                new_predictions[key] = []
            new_predictions[key].append(value)

    results['predictions'] = new_predictions

    results['fit_time'] = np.array(results['fit_time'])
    results['score_time'] = np.array(results['score_time'])
    results['test_score'] = np.array(results['test_score'])
    results['feature_importance'] = np.array(results['feature_importance'])
