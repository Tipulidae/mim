from keras import Model, Input
from keras.layers import Dense, Flatten

from mim.util.logs import get_logger

log = get_logger('simple_nn')


def basic_ff():
    inp = Input(shape=(128, ))
    x = Flatten()(inp)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inp, output)
    return model
