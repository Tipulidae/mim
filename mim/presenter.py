import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_auc_score
)

from mim.util.logs import get_logger
from mim.util.util import ranksort
from mim.experiments.experiments import result_path
from mim.experiments.factory import experiment_from_name

log = get_logger("Presenter")


class Presenter:
    def __init__(self, name):
        self.results = dict()
        log.debug(f'Loading all experiments for {name}')

        self.experiments = experiment_from_name(name)
        for xp in self.experiments:
            path = result_path(xp)
            try:
                self.results[xp.name] = pd.read_pickle(path)
            except FileNotFoundError:
                log.debug(f"Test {xp.name} doesn't exist in path {path}")

    def describe(self, like='.*'):
        results = []
        for name, xp in self._results_that_match_pattern(like):
            results.append(pd.Series(
                data=[
                    np.mean(xp['test_score']),
                    np.std(xp['test_score']),
                    np.mean(xp['train_score']),
                    xp['metadata']['current_commit'][:8],
                    xp['metadata']['has_uncommitted_changes'],
                    xp['metadata']['timestamp']],
                index=[
                    'test_score',
                    'test_score_std',
                    'train_score',
                    'commit',
                    'changed',
                    'timestamp'],
                name=name))
        return pd.DataFrame(results)

    def train_test_scores(self, name):
        xp = self.results[name]
        return pd.DataFrame(
            [np.array(xp['train_score']), xp['test_score']],
            index=['train', 'test']
        )

    def scores(self, like='.*'):
        results = []
        for name, xp in self._results_that_match_pattern(like):
            targets, predictions = self._target_predictions(xp)
            results.append(pd.Series(
                data=[
                    roc_auc_score(targets, predictions)
                ],
                index=[
                    'auc'
                ],
                name=name))
        return pd.DataFrame(results)

    def threshold_scores(self, like='.*', threshold=0.5):
        results = []
        for name, xp in self._results_that_match_pattern(like):
            targets, predictions = self._threshold_target_predictions(
                xp, threshold)
            results.append(pd.Series(
                data=[
                    precision_score(targets, predictions),
                    recall_score(targets, predictions),
                    accuracy_score(targets, predictions),
                    f1_score(targets, predictions)
                ],
                index=[
                    'precision',
                    'recall',
                    'accuracy',
                    'f1'
                ],
                name=name))
        return pd.DataFrame(results)

    def confusion_matrix(self, name, threshold=0.5, normalize=None):
        targets, predictions = self._threshold_target_predictions(
            self.results[name], threshold
        )
        return confusion_matrix(targets, predictions, normalize=normalize)

    def feature_importance(self, name):
        """
        Most important feature has the smallest value (ie the highest rank)
        :param name: Name of experiment
        :return: Series with feature names as index and average feature
        importance rank (lower is more important) as values.
        """
        return self.feature_importance_rank_df(name).mean().sort_values()

    def feature_importance_df(self, name):
        xp = self.results[name]
        assert (fi := xp['feature_importance']) is not None
        return pd.DataFrame([f for f in fi], columns=xp['feature_names'])

    def feature_importance_rank_df(self, name):
        xp = self.results[name]
        assert (fi := xp['feature_importance']) is not None
        return pd.DataFrame(
            [ranksort(f, ascending=False) for f in fi],
            columns=xp['feature_names']
        )

    def _results_that_match_pattern(self, pattern):
        p = re.compile(pattern)
        for name in filter(p.match, self.results):
            yield name, self.results[name]

    def _threshold_target_predictions(self, xp, threshold):
        targets, predictions = self._target_predictions(xp)
        predictions = (predictions > threshold).astype(int)
        return targets, predictions

    def _target_predictions(self, xp):
        predictions = pd.concat(xp['predictions']['prediction'], axis=0)
        targets = pd.DataFrame(np.concatenate(xp['targets']))
        return targets, predictions

    def _is_classifier(self, xps):
        for xp in xps:
            if not self.experiments[xp].algorithm.model_type.is_classification:
                print(f'Experiment {xp} is not a classification problem!')
                return False

        return True

    def _has_same_date_range(self, ts):
        first = self._get_validation_dates(ts[0])
        for t in ts:
            if len(self._get_validation_dates(t)) != len(first):
                return False

        return True

    def _get_validation_dates(self, test_name):
        cv = self.experiments[test_name].cross_validation
        log.debug(test_name)
        return range(cv.n_splits)

    def _is_loaded(self, ts):
        for t in ts:
            if not (self._is_valid_test_case(t) and self.results[t]):
                print(f"Test case {t} doesn't exist or isn't loaded!")
                return False

        return True

    def _is_valid_test_case(self, t):
        return t in self.results
