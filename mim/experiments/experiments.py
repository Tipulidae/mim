import os
from copy import copy
from time import time
from pathlib import Path
from typing import Any, NamedTuple, Callable, Union

import numpy as np
import pandas as pd
# noinspection PyUnresolvedReferences
import silence_tensorflow.auto  # noqa: F401
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

from mim.extractors.extractor import Extractor
from mim.cross_validation import CrossValidationWrapper, ChronologicalSplit
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
    index: Any = None
    features: Any = None
    labels: Any = None
    post_processing: Any = None
    model: Any = RandomForestClassifier
    model_kwargs: dict = {}
    building_model_requires_development_data: bool = False
    optimizer: Any = 'adam',
    loss: Any = 'binary_crossentropy'
    epochs: Union[int, hp.Param] = None
    batch_size: Union[int, hp.Param] = 64
    metrics: Any = ['accuracy', 'auc']
    ignore_callbacks: bool = False
    random_state: int = 123
    cv: Any = KFold
    cv_kwargs: dict = {}
    scoring: Any = roc_auc_score
    hold_out: Any = ChronologicalSplit
    hold_out_size: float = 0.25
    log_conda_env: bool = True
    alias: str = ''
    parent_base: str = None
    parent_name: str = None

    def run(self):
        self._run()
        try:
            results = self._run()
            pd.to_pickle(results, self.result_path)
            log.debug(f'Saved results in {self.result_path}')
        except Exception as e:
            log.error('Something went wrong!')
            raise e

    def _run(self):
        log.info(f'Running experiment {self.name}: {self.description}')
        t = time()

        data, hold_out = self.get_data()
        cv = self.cross_validation

        # TODO:
        #  Add feature names to dataset somewhere so it can be
        #  logged here
        feature_names = None

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

        for i, (train, validation) in enumerate(cv.split(data)):
            result = _validate(
                train,
                validation,
                self.get_model(train, validation),
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

    def get_data(self):
        """
        Uses the extractor and specifications to create the X and y data
        set.

        :return: Data object
        """
        specification = {
            'index': self.index,
            'features': self.features,
            'labels': self.labels,
            'processing': self.post_processing,
        }
        data = self.extractor(**specification).get_data()
        splitter = self.hold_out(test_size=self.hold_out_size)
        develop_index, test_index = next(splitter.split(data))
        return data.split(develop_index, test_index)

    def get_model(self, train, validation):
        model_kwargs = copy(self.model_kwargs)

        if self.building_model_requires_development_data:
            model_kwargs['train'] = train
            model_kwargs['validation'] = validation

        # Releases keras global state. Ref:
        # https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
        tf.keras.backend.clear_session()
        model = self.model(**model_kwargs)

        if isinstance(model, tf.keras.Model):
            if isinstance(self.optimizer, dict):
                optimizer = self.optimizer['name'](**self.optimizer['kwargs'])
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
                checkpoint_path=self.base_path,
                tensorboard_path=self.base_path,
                random_state=self.random_state,
                batch_size=self.batch_size,
                epochs=self.epochs,
                optimizer=optimizer,
                loss=self.loss,
                metrics=metric_list,
                ignore_callbacks=self.ignore_callbacks
            )
        else:
            model.random_state = self.random_state
            return Model(model)

    def asdict(self):
        return callable_to_string(self._asdict())

    @property
    def cross_validation(self):
        return CrossValidationWrapper(self.cv, **self.cv_kwargs)

    @property
    def is_binary(self):
        return self.labels['is_binary']

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


def _validate(train, val, model, scoring, split_number=None):
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
        'targets': y_val,
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
