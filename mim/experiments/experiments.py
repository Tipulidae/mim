import os
from pathlib import Path
from typing import Any, NamedTuple, Callable

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

from mim.extractors.extractor import Extractor
from mim.cross_validation import CrossValidationWrapper, ChronologicalSplit
from mim.config import PATH_TO_TEST_RESULTS
from mim.model_wrapper import Model, KerasWrapper


class Experiment(NamedTuple):
    description: str
    alias: str = None
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
    epochs: int = None
    batch_size: int = 64
    metrics: Any = ['accuracy', 'auc']
    ignore_callbacks: bool = False
    random_state: int = 123
    cv: Any = KFold
    cv_kwargs: dict = {}
    scoring: Any = roc_auc_score
    hold_out: Any = ChronologicalSplit
    hold_out_size: float = 0

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
        model_kwargs = self.model_kwargs

        if self.building_model_requires_development_data:
            model_kwargs['train'] = train
            model_kwargs['validation'] = validation

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
                xp_name=self.name,
                xp_class=self.__class__.__name__,
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
            return Model(
                model,
                xp_name=self.name,
                xp_class=self.__class__.__name__,
            )

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
            PATH_TO_TEST_RESULTS,
            f'{self.__class__.__name__}_{self.name}.experiment')

    @property
    def is_done(self):
        file = Path(self.result_path)
        return file.is_file()


def result_path(xp):
    return os.path.join(
        PATH_TO_TEST_RESULTS,
        f'{xp.__class__.__name__}_{xp.name}.experiment')
