import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, roc_curve
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy


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


def total_confusion(targets, predictions):
    # Create a dataframe containing the confusion matrix (and tpr, fpr, npv
    # and ppv) for all unique thresholds.
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    metrics = pd.DataFrame(
        [fpr, tpr],
        index=['fpr', 'tpr'],
        columns=pd.Index(thresholds, name='threshold')
    ).T
    positives = targets.sum()
    negatives = len(targets) - positives
    metrics['tp'] = tpr * positives
    metrics['fp'] = fpr * negatives
    metrics['fn'] = positives - metrics.tp
    metrics['tn'] = negatives - metrics.fp

    metrics['npv'] = metrics.tn / (metrics.tn + metrics.fn)
    metrics['ppv'] = metrics.tp / (metrics.tp + metrics.fp)
    return metrics


def rule_in_rule_out(
        targets,
        predictions,
        rule_in_spec=0.90,
        rule_in_ppv=0.7,
        rule_out_sens=0.99,
        rule_out_npv=0.995,
        rule_in_threshold=None,
        rule_out_threshold=None):
    """
    The 'naive' implementation of rule-in rule-out. Calculates a 3*n numpy
    array where the columns correspond to rule-in, intermediate and rule-out,
    respectively. There might not be a threshold that satisfies the
    constraints, in which case the corresponding group (rule-in or rule-out)
    is empty.
    """
    if rule_in_threshold is None or rule_out_threshold is None:
        in_th, out_th = find_rule_in_rule_out_thresholds(
            targets, predictions, rule_in_spec=rule_in_spec,
            rule_in_ppv=rule_in_ppv, rule_out_sens=rule_out_sens,
            rule_out_npv=rule_out_npv
        )
        if rule_in_threshold is None:
            rule_in_threshold = in_th
        if rule_out_threshold is None:
            rule_out_threshold = out_th

    rule_in = np.where(predictions >= rule_in_threshold, 1, 0)
    rule_out = np.where(predictions < rule_out_threshold, 1, 0)
    results = np.vstack(
        [rule_in, np.ones_like(rule_out) - rule_out - rule_in, rule_out]).T

    return results


def find_rule_in_rule_out_thresholds(
        targets,
        predictions,
        rule_in_spec=0.90,
        rule_in_ppv=0.7,
        rule_out_sens=0.99,
        rule_out_npv=0.995):
    metrics = total_confusion(targets, predictions)

    # specificity == 1 - fpr
    rule_in = metrics[
        ((1 - metrics.fpr) >= rule_in_spec) & (metrics.ppv >= rule_in_ppv)]
    if len(rule_in) > 0:
        rule_in_threshold = rule_in.index[-1]
    else:
        rule_in_threshold = 1.0

    # sensitivity == tpr
    rule_out = metrics[
        (metrics.tpr >= rule_out_sens) & (metrics.npv >= rule_out_npv)]
    if len(rule_out) > 0:
        rule_out_threshold = rule_out.index[0]
    else:
        rule_out_threshold = 0.0

    return rule_in_threshold, rule_out_threshold


def rule_in(targets, predictions, rule_in_spec=0.90, rule_in_ppv=0.7):
    results = rule_in_rule_out(
        targets,
        predictions,
        rule_in_spec=rule_in_spec,
        rule_in_ppv=rule_in_ppv
    )
    return results.mean(axis=0)[0]


def rule_out(targets, predictions, rule_out_sens=0.99, rule_out_npv=0.995):
    if isinstance(targets, pd.DataFrame):
        targets = targets.values.ravel()
    if isinstance(predictions, pd.DataFrame):
        predictions = predictions.values.ravel()
    if len(predictions.shape) > 1:
        predictions = predictions.ravel()

    results = rule_in_rule_out(
        targets,
        predictions,
        rule_out_sens=rule_out_sens,
        rule_out_npv=rule_out_npv
    )
    return results.mean(axis=0)[-1]


def rule_in_rule_out_ab(
        targets,
        predictions,
        rule_in_spec=0.90,
        rule_in_ppv=0.7,
        rule_out_sens=0.99,
        rule_out_npv=1.0
):
    # Some copy-paste from AB's rule-in/rule-out calculations, mainly so
    # that I can compare and make sure that it agrees with my implementation.
    def prob_label_matrix_to_thresholds(plm):
        plm = plm[plm[:, 0].argsort()]
        thresholds = [
            (plm[split_point, 0] + plm[split_point + 1, 0]) / 2
            for split_point in range(len(plm) - 1)
        ]
        thresholds = sorted(set(filter(lambda x: x < 1, thresholds)))
        return thresholds

    def eval_threshold(th, plm):
        neg_pred = plm[plm[:, 0] <= th]
        pos_pred = plm[plm[:, 0] > th]
        if len(pos_pred) == 0:
            print("Empty pos-pred with th: %f" % th)

        tp = np.sum(pos_pred[:, 1])
        fn = np.sum(neg_pred[:, 1])
        tn = len(neg_pred) - fn
        fp = len(pos_pred) - tp
        assert fn + tn + tp + fp == len(plm)

        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        return [tp, tn, fp, fn, sensitivity, specificity, ppv, npv]

    def find_best_riro_ths(plm, min_sens, min_npv, min_ppv,
                           min_spec, quiet=False, thresholds=None):
        best_rule_out = None
        best_rule_in = None
        if not thresholds:
            thresholds = prob_label_matrix_to_thresholds(plm)
        for th in thresholds:
            r = eval_threshold(th, plm)
            out = r + [th]
            if not quiet:
                print("\t".join(map(str, out)))
            if r[4] >= min_sens and r[7] >= min_npv:
                best_rule_out = out
            if r[6] >= min_ppv and r[5] >= min_spec and best_rule_in is None:
                best_rule_in = out
        if best_rule_in is None:
            best_rule_in = out
        return best_rule_out, best_rule_in

    prob_label_matrix = np.vstack([predictions, targets]).T
    return find_best_riro_ths(
        prob_label_matrix,
        min_sens=rule_out_sens,
        min_npv=rule_out_npv,
        min_ppv=rule_in_ppv,
        min_spec=rule_in_spec,
        quiet=True
    )


def sparse_categorical_accuracy(y_true, y_pred):
    m = SparseCategoricalAccuracy()
    m.update_state(y_true, y_pred)
    return m.result().numpy()
