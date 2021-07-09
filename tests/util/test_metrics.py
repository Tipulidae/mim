import numpy as np

from mim.util.metrics import rule_in_confusion, rule_out_confusion


def test_rule_in_rule_out_confusion():
    predictions = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    targets = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0])

    tn, fp, fn, tp = rule_in_confusion(targets, predictions, threshold=0.75)
    assert [tn, fp, fn, tp] == [5, 1, 3, 2]

    tn, fp, fn, tp = rule_out_confusion(targets, predictions, threshold=0.75)
    assert [tn, fp, fn, tp] == [2, 3, 1, 5]
