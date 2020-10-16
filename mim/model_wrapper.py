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
            transformer=None,
            transformer_args=None,
            wrapper=None,
            **kwargs):
        if transformer:
            self.transformer = transformer(**transformer_args)
        else:
            self.transformer = None

        self.model = model(*args, **kwargs)
        if wrapper is not None:
            self.model = wrapper(self.model)

    def predict(self, x):
        result = {}
        if isinstance(x, pd.DataFrame):
            index = x.index
            x = x.values
        else:
            index = None

        prediction = self._prediction(x)

        if self.only_last_prediction_column_is_used:
            prediction = prediction[:, 1]

        if self.transformer:
            prediction = self.transformer.inverse_transform(
                prediction.reshape(-1, 1))

        result['prediction'] = pd.DataFrame(prediction, index=index)
        return result

    def fit(self, x, y, **kwargs):
        y = _numpy(y)
        x = _numpy(x)

        if self.transformer:
            self.transformer.fit(y)
            y = self.transformer.transform(y)

        self.model.fit(x, y.ravel(), **kwargs)

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
        super().__init__(ensemble.RandomForestClassifier, *args,
                         random_state=random_state, **kwargs)


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
            **kwargs):
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        super().__init__(model, *args, **kwargs)
        self.model.compile()
        self._history = None

    def fit(self, x, y, validation_data=None):
        self._history = self.model.fit(
            x=x,
            y=y,
            validation_data=validation_data,
        )

    @property
    def history(self):
        return self._history.history

    @property
    def only_last_prediction_column_is_used(self):
        return False

    def _prediction(self, x):
        return self.model.predict(x)
