import os
from enum import Enum

import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from mim.config import PATH_TO_TF_LOGS, PATH_TO_TF_CHECKPOINTS
from mim.util.logs import get_logger

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
            *args,
            xp_name=None,
            xp_class=None,
            can_use_tf_dataset=False,
            **kwargs):
        self.can_use_tf_dataset = can_use_tf_dataset
        self.model = model(*args, **kwargs)
        self.xp_name = xp_name
        self.xp_class = xp_class

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

    def fit(self, data, validation_data=None, **kwargs):
        if self.can_use_tf_dataset:
            train = prepare_dataset(data, prefetch=3, **kwargs)
            val = prepare_dataset(validation_data, prefetch=3, **kwargs)
            return self.model.fit(train, validation_data=val, **kwargs).history
        else:
            x = data['x'].as_numpy
            y = data['y'].as_numpy.ravel()
            return self.model.fit(x, y)

    @property
    def only_last_prediction_column_is_used(self):
        return (self.model_type is ModelTypes.CLASSIFIER and
                len(self.model.classes_) == 2)

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
            model: tf.keras.Model,
            *args,
            random_state=42,
            batch_size=16,
            epochs=2,
            compile_args=None,
            ignore_callbacks=False,
            **kwargs):
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        super().__init__(model, *args, can_use_tf_dataset=True, **kwargs)
        if compile_args is None:
            compile_args = {
                'optimizer': tf.keras.optimizers.Adam(1e-4),
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy', tf.keras.metrics.AUC()]
            }

        self.model.compile(**compile_args)
        self.batch_size = batch_size
        self.epochs = epochs
        self.ignore_callbacks = ignore_callbacks
        log.info(self.model.summary())

    def fit(self, data, validation_data=None, **kwargs):
        checkpoint_path = os.path.join(
            PATH_TO_TF_CHECKPOINTS,
            self.xp_class,
            self.xp_name
        )
        tensorboard_path = os.path.join(
            PATH_TO_TF_LOGS,
            self.xp_class,
            self.xp_name
        )
        if self.ignore_callbacks:
            callbacks = None
        else:
            callbacks = [
                ModelCheckpoint(
                    filepath=os.path.join(checkpoint_path, 'last.ckpt')
                ),
                ModelCheckpoint(
                    filepath=os.path.join(checkpoint_path, 'best.ckpt'),
                    save_best_only=True
                ),
                TensorBoard(log_dir=tensorboard_path)
            ]
        return super().fit(
            data,
            validation_data=validation_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            **kwargs
        )

    @property
    def only_last_prediction_column_is_used(self):
        return False

    def _prediction(self, x):
        return self.model.predict(x.batch(self.batch_size))


def prepare_dataset(data, batch_size=1, prefetch=None, **kwargs):
    x = data['x'].as_dataset
    y = data['y'].as_dataset

    fixed_data = tf.data.Dataset.zip((x, y)).batch(batch_size)
    if prefetch:
        fixed_data = fixed_data.prefetch(prefetch)

    return fixed_data
