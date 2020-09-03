from enum import Enum

import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model


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

    def predict(self, X):
        result = {}

        prediction = self._prediction(X)

        if self.only_last_prediction_column_is_used:
            prediction = prediction[:, 1]

        if self.transformer:
            prediction = self.transformer.inverse_transform(
                prediction.reshape(-1, 1))

        result['prediction'] = pd.DataFrame(prediction, index=X.index)
        return result

    def fit(self, X, y):
        y = y.values
        if self.transformer:
            self.transformer.fit(y)
            y = self.transformer.transform(y)

        if isinstance(X, pd.DataFrame):
            X = X.values

        self.model.fit(X, y.ravel())

    @property
    def only_last_prediction_column_is_used(self):
        return (self.model_type is ModelTypes.CLASSIFIER and
                len(self.model.classes_) == 2)

    @property
    def history(self):
        return None

    def _prediction(self, X):
        return self.model.predict_proba(X.values)


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

    def _prediction(self, X):
        return self.model.predict(X)


class RandomForestRegressor(Model):
    model_type = ModelTypes.REGRESSOR

    def __init__(self, *args, random_state=126, **kwargs):
        super().__init__(ensemble.RandomForestRegressor, *args,
                         random_state=random_state, **kwargs)

    def _prediction(self, X):
        return self.model.predict(X)


class NullModel:
    classes_ = []

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        return None
