import os
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
from mim.util.util import ranksort, insensitive_iglob
from mim.experiments.hyper_parameter import flatten
from mim.config import PATH_TO_TEST_RESULTS

log = get_logger("Presenter")


class Presenter:
    def __init__(self, name):
        self.results = dict()
        paths = insensitive_iglob(
            f"{PATH_TO_TEST_RESULTS}/{name}/**/results.pickle",
            recursive=True
        )

        for path in sorted(paths):
            _, xp_name = os.path.split(os.path.dirname(path))
            log.info(f"Loading {xp_name}")
            if xp_name in self.results:
                log.warning(f"Two experiments with the name {xp_name}!")

            self.results[xp_name] = pd.read_pickle(path)

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

    def summary(self):
        flat_results = [
            pd.Series(flatten(xp['experiment_summary']), name=name)
            for name, xp in self.results.items() if 'experiment_summary' in xp
        ]
        df = pd.concat(flat_results, axis=1)
        return df

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

    def history(self, name):
        xp = self.results[name]
        if xp['history'] is None:
            print(f"Experiment {name} has no history.")
            return

        history = pd.concat(
            [pd.DataFrame(h) for h in xp['history']],
            axis=1,
            keys=[f'fold {i}' for i in range(len(xp['history']))]
        )
        return history
        # history.plot()

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
            if not self.experiments[xp].model.model_type.is_classification:
                print(f'Experiment {xp} is not a classification problem!')
                return False

        return True

    def _is_loaded(self, ts):
        for t in ts:
            if not (self._is_valid_test_case(t) and self.results[t]):
                print(f"Test case {t} doesn't exist or isn't loaded!")
                return False

        return True

    def _is_valid_test_case(self, t):
        return t in self.results
