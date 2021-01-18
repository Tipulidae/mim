from enum import Enum

import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import tensorflow as tf


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
            *args,
            can_use_tf_dataset=False,
            **kwargs):
        self.can_use_tf_dataset = can_use_tf_dataset
        self.model = model(*args, **kwargs)

    def predict(self, x):
        result = {}
        if self.can_use_tf_dataset:
            x = x.as_dataset
        else:
            x = x.as_numpy
        prediction = self._prediction(x)

        if self.only_last_prediction_column_is_used:
            prediction = prediction[:, 1]

        result['prediction'] = pd.DataFrame(prediction)
        return result

    def fit(self, data, **kwargs):
        if self.can_use_tf_dataset:
            self.model.fit(data.as_dataset, **kwargs)
        else:
            x = data['x'].as_numpy
            y = data['y'].as_numpy.ravel()
            self.model.fit(x, y, **kwargs)

    @property
    def only_last_prediction_column_is_used(self):
        return (self.model_type is ModelTypes.CLASSIFIER and
                len(self.model.classes_) == 2)

    @property
    def history(self):
        return None

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
            ensemble.RandomForestClassifier,
            *args,
            can_use_tf_dataset=False,
            random_state=random_state,
            **kwargs
        )


class ExtraTreesClassifier(Model):
    def __init__(self, *args, random_state=124, **kwargs):
        super().__init__(ensemble.ExtraTreesClassifier, *args,
                         random_state=random_state, **kwargs)


class GradientBoostingClassifier(Model):
    def __init__(self, *args, random_state=125, **kwargs):
        super().__init__(ensemble.GradientBoostingClassifier, *args,
                         random_state=random_state, **kwargs)


class LogisticRegression(Model):
    def __init__(self, *args, random_state=125, **kwargs):
        super().__init__(linear_model.LogisticRegression, *args,
                         random_state=random_state, **kwargs)


class LinearRegression(Model):
    model_type = ModelTypes.REGRESSOR

    def __init__(self, *args, **kwargs):
        super().__init__(linear_model.LinearRegression, *args, **kwargs)

    def _prediction(self, x):
        return self.model.predict(x)


class RandomForestRegressor(Model):
    model_type = ModelTypes.REGRESSOR

    def __init__(self, *args, random_state=126, **kwargs):
        super().__init__(ensemble.RandomForestRegressor, *args,
                         random_state=random_state, **kwargs)

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


class KerasWrapper(Model):
    def __init__(
            self,
            model,
            *args,
            random_state=42,
            batch_size=16,
            epochs=2,
            compile_args=None,
            **kwargs):
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        super().__init__(model, *args, can_use_tf_dataset=True, **kwargs)
        if compile_args is None:
            compile_args = {
                'optimizer': 'sgd',
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy']
            }

        self.model.compile(**compile_args)
        self._history = None
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, data, validation_data=None):
        x = data['x'].as_dataset
        y = data['y'].as_dataset

        self._history = self.model.fit(
            x=tf.data.Dataset.zip((x, y)).batch(self.batch_size).prefetch(3),
            validation_data=validation_data,
            epochs=self.epochs
        )

    @property
    def history(self):
        return self._history.history

    @property
    def only_last_prediction_column_is_used(self):
        return False

    def _prediction(self, x):
        return self.model.predict(x.batch(self.batch_size))
