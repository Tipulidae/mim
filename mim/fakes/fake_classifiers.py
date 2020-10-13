import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from mim.model_wrapper import Model, ModelTypes


class RandomRegressor(Model):
    """
    Use this to make random predictions in different ways. There are three
    modes:

    mode == 'mean':
        prediction is the mean of y_train
    mode == 'distribution':
        prediction is sampled from distribution of y_train
    else:
        prediction is sampled from U(0, 1)
    """
    model_type = ModelTypes.REGRESSOR

    def __init__(self, *args, **kwargs):
        super().__init__(self.Inner, *args, **kwargs)

    class Inner:
        """
        Ok, so this might seem a bit 'ungodly' at first (I agree it's not
        optimal). The thing is, to inherit from Model, like RandomRegressor
        does, it needs to send an object that fulfills certain methods.
        In this case, we want a very particular (random) behavior for those
        methods. That's the point of this Inner class. We could certainly
        move this outside of RandomRegressor, but since it's only ever used
        here, it feels like it belongs inside RandomRegressor.

        If you do RandomRegressor().model, you would get the instance of
        this Inner object.
        """

        def __init__(self, mode='mean', random_state=42):
            # Each instance of this class uses it's own 'private' random state
            self.r = np.random.RandomState(random_state)
            self.mode = mode.lower()
            self.y = None

        def fit(self, X, y):
            self.y = y

        def predict_proba(self, X):
            if self.mode == 'mean':
                return len(X)*[np.mean(self.y)]
            elif self.mode == 'distribution':
                return self.r.choice(self.y, size=len(X), replace=True)
            else:
                return self.r.rand(len(X))


class Ann(keras.Sequential):
    def __init__(self):
        super().__init__(
            layers=[
                layers.Flatten(input_shape=(784,), name='digits'),
                layers.Dense(64, activation='relu', name='dense_1'),
                layers.Dense(64, activation='relu', name='dense_2'),
                layers.Dense(10, activation='softmax', name='predictions')
            ]
        )

    def compile(self, **kwargs):
        super().compile(
            optimizer='rmsprop',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            **kwargs
        )

    def fit(self, **kwargs):
        return super().fit(
            batch_size=64,
            epochs=2,
            # callbacks=[
            #     keras.callbacks.TensorBoard(log_dir=PATH_TO_TEST_RESULTS)
            # ],
            **kwargs
        )
