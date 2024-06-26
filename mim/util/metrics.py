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


def total_confusion_all_thresholds(targets, predictions):
    # The original total_confusion cleverly discards some of the thresholds
    # that don't matter for fpr and tpr calculations. This functions creates
    # a similar table of tp, fp, tn, fn columns, but for all the predictions,
    # including 0 and 1.
    cm = pd.DataFrame(
        np.concatenate([[0], _to_vector(targets), [0]]),
        index=pd.Index(
            np.concatenate([[0.0], _to_vector(predictions), [1.0]]),
            name='threshold'),
        columns=['y']
    ).sort_index()
    cm['tn'] = (1 - cm.y).cumsum() - 1
    cm.tn.iloc[-1] -= 1
    cm['fn'] = cm.y.cumsum()
    cm['tp'] = cm.y.sum() - cm.fn
    cm['fp'] = len(cm) - 2 - cm.y.sum() - cm.tn
    return cm[['tn', 'fn', 'tp', 'fp']]


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

    ruled_in = np.where(predictions >= rule_in_threshold, 1, 0)
    ruled_out = np.where(predictions < rule_out_threshold, 1, 0)
    results = np.vstack(
        [ruled_in,
         np.ones_like(ruled_out) - ruled_out - ruled_in,
         ruled_out]
    ).T

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
    ruled_in = metrics[
        ((1 - metrics.fpr) >= rule_in_spec) & (metrics.ppv >= rule_in_ppv)]
    if len(ruled_in) > 0:
        rule_in_threshold = ruled_in.index[-1]
    else:
        rule_in_threshold = 1.0

    # sensitivity == tpr
    ruled_out = metrics[
        (metrics.tpr >= rule_out_sens) & (metrics.npv >= rule_out_npv)]
    if len(ruled_out) > 0:
        rule_out_threshold = ruled_out.index[0]
    else:
        rule_out_threshold = 0.0

    return rule_in_threshold, rule_out_threshold


def rule_in(targets, predictions, rule_in_spec=0.90, rule_in_ppv=0.7,
            threshold=None):
    targets = _to_vector(targets)
    predictions = _to_vector(predictions)
    results = rule_in_rule_out(
        targets,
        predictions,
        rule_in_spec=rule_in_spec,
        rule_in_ppv=rule_in_ppv,
        rule_in_threshold=threshold
    )
    return results.mean(axis=0)[0]


def rule_out(targets, predictions, rule_out_sens=0.99, rule_out_npv=0.995,
             threshold=None):
    targets = _to_vector(targets)
    predictions = _to_vector(predictions)

    results = rule_in_rule_out(
        targets,
        predictions,
        rule_out_sens=rule_out_sens,
        rule_out_npv=rule_out_npv,
        rule_out_threshold=threshold
    )
    return results.mean(axis=0)[-1]


def _to_vector(data):
    if isinstance(data, pd.DataFrame):
        return data.values.ravel()
    if isinstance(data, pd.Series):
        return data.values
    if len(data.shape) > 1:
        return data.ravel()
    return data


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


def max_mcc(targets, predictions):
    """
    Calculates Matthew's Correlation Coefficient (mcc) for all possible
    thresholds, and returns the largets threshold + corresponding mcc.
    For given threshold t, uses y = np.where(pred > th, 1, 0) as the
    predictions.
    """
    tc = total_confusion_all_thresholds(targets, predictions)
    tc['mcc'] = (
        (tc.tp * tc.tn - tc.fp * tc.fn) /
        (
            ((tc.tp + tc.fp) * (tc.fn + tc.tn) *
             (tc.tp + tc.fn) * (tc.fp + tc.tn)) ** 0.5
        )
    )
    return tc.mcc.idxmax(), tc.mcc.max()
