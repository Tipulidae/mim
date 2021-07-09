import numpy as np

from sklearn.metrics import confusion_matrix


def positive_predictive_value(targets, predictions):
    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
    return tp / (tp + fp)


def negative_predictive_value(targets, predictions):
    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
    return tn / (tn + fn)


def rule_in_confusion(targets, predictions, threshold=0.5):
    predictions = np.where(predictions >= threshold, 1, 0)
    return confusion_matrix(targets, predictions).ravel()


def rule_out_confusion(targets, predictions, threshold=0.5):
    tp, fn, fp, tn = rule_in_confusion(targets, predictions,
                                       threshold=threshold)
    return tn, fp, fn, tp
