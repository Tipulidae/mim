from tensorflow.keras.metrics import SparseCategoricalAccuracy


def sparse_categorical_accuracy(y_true, y_pred):
    m = SparseCategoricalAccuracy()
    m.update_state(y_true, y_pred)
    return m.result().numpy()
