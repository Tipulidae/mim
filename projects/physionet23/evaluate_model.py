import numpy as np
import pandas as pd


def new_challenge_score(targets, predictions):
    return challenge_score(
        _to_vector(targets),
        _to_vector(predictions)
    )


def _to_vector(data):
    if isinstance(data, pd.DataFrame):
        return data.values.ravel()
    if len(data.shape) > 1:
        return data.ravel()
    return data


def challenge_score(labels, outputs):
    # print(f"{labels=}, {outputs=}")
    assert len(labels) == len(outputs)
    num_instances = len(labels)

    # Use the unique output values as the thresholds for the positive and
    # negative classes.
    thresholds = np.unique(outputs)
    thresholds = np.append(thresholds, thresholds[-1]+1)
    thresholds = thresholds[::-1]
    num_thresholds = len(thresholds)

    idx = np.argsort(outputs)[::-1]

    # Initialize the TPs, FPs, FNs, and TNs with no positive outputs.
    tp = np.zeros(num_thresholds)
    fp = np.zeros(num_thresholds)
    fn = np.zeros(num_thresholds)
    tn = np.zeros(num_thresholds)

    tp[0] = 0
    fp[0] = 0
    fn[0] = np.sum(labels == 1)
    tn[0] = np.sum(labels == 0)

    # Update the TPs, FPs, FNs, and TNs using the values at the previous
    # threshold.
    i = 0
    for j in range(1, num_thresholds):
        tp[j] = tp[j-1]
        fp[j] = fp[j-1]
        fn[j] = fn[j-1]
        tn[j] = tn[j-1]

        while i < num_instances and outputs[idx[i]] >= thresholds[j]:
            if labels[idx[i]]:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1

    # Compute the TPRs and FPRs.
    tpr = np.zeros(num_thresholds)
    fpr = np.zeros(num_thresholds)
    for j in range(num_thresholds):
        if tp[j] + fn[j] > 0:
            tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            fpr[j] = float(fp[j]) / float(fp[j] + tn[j])
        else:
            tpr[j] = float('nan')
            fpr[j] = float('nan')

    # Find the largest TPR such that FPR <= 0.05.
    max_fpr = 0.05
    max_tpr = float('nan')
    if np.any(fpr <= max_fpr):
        indices = np.where(fpr <= max_fpr)
        max_tpr = np.max(tpr[indices])

    return max_tpr
