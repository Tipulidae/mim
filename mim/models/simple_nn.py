from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten

from mim.util.logs import get_logger

log = get_logger('simple_nn')


def basic_ff():
    inp = Input(shape=(128, ))
    x = Flatten()(inp)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = keras.Model(inp, output)
    return model
