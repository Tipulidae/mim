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
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt

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
        return df.T.join(self.scores())

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

    def predictions(self, like='.*'):
        # Return dataframe with the true targets and predictions for each
        # experiment
        all_predictions = []
        all_targets = []
        for name, xp in self._results_that_match_pattern(like):
            targets, predictions = self._target_predictions(xp)
            all_targets.append(targets.values)
            all_predictions.append(
                predictions.iloc[:, 0].rename(name)
            )
        assert all_columns_equal(np.concatenate(all_targets, axis=1))

        target = pd.DataFrame(all_targets[0], columns=['y'])
        return target.join(pd.DataFrame(all_predictions).T)

    def prediction_ranks(self, like='.*'):
        predictions = self.predictions(like=like)
        for col in predictions.columns[1:]:
            predictions.loc[:, col] = ranksort(predictions.loc[:, col])

        predictions.iloc[:, 1:] /= len(predictions)
        return predictions

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

    def roc(self, xps, figsize=(20, 20)):
        """
        Plot roc-curves for each of the experiments in xps.

        :param xps: list of names of experiments
        :param figsize: Figure size
        """
        plt.figure(figsize=figsize)
        labels = []
        lines = []
        for xp in xps:
            r = self.results[xp]
            targets = np.concatenate(r['targets'])
            predictions = pd.concat(r['predictions']['prediction'])
            fpr, tpr, thresholds = roc_curve(targets, predictions)
            auc = roc_auc_score(targets, predictions)
            l, = plt.plot(fpr, tpr, lw=1, alpha=1)
            lines.append(l)
            labels.append(f'{xp} - AUC = {auc:.4f}')

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(lines, labels)
        plt.grid(which='both')

        df = pd.DataFrame(
            [self.results[xp]['test_score'] for xp in xps],
            index=xps,
        ).T
        plt.show()
        return df

    def sensitivity_specificity(self, experiments):
        """
        Return DataFrame containing the sensitivity and specificity of each
        experiment, for each threshold. The dataframe is cropped to
        100 rows.

        :param experiments: list of names of experiments
        """
        def sample_evenly(x, n):
            return x[::len(x) // n][:n]

        output = []
        for experiment in experiments:
            r = self.results[experiment]
            targets = np.concatenate(r['targets'])
            predictions = pd.concat(r['predictions']['prediction'])
            fpr, tpr, _ = roc_curve(targets, predictions)

            sensitivity = sample_evenly(tpr, 100)
            specificity = 1 - sample_evenly(fpr, 100)

            data = np.array([sensitivity, specificity]).T
            output.append(pd.DataFrame(data))

        output = pd.concat(output, axis=1)
        output.columns = pd.MultiIndex.from_product(
            [experiments, ['Sensitivity', 'Specificity']])
        return output

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

    def history(self, name, columns='all', folds='all'):
        xp = self.results[name]
        if xp['history'] is None:
            print(f"Experiment {name} has no history.")
            return

        history = pd.concat(
            [pd.DataFrame(h) for h in xp['history']],
            axis=1,
            keys=[f'fold {i}' for i in range(len(xp['history']))]
        )

        if isinstance(columns, list):
            history = history.loc[:, pd.IndexSlice[:, columns]]
        if isinstance(folds, list):
            fold_names = [f'fold {i}' for i in folds]
            history = history.loc[:, pd.IndexSlice[fold_names, :]]
        if folds == 'first':
            history = history.loc[:, 'fold 0']

        return history

    def plot_history(self, names, columns=None,
                     folds='first', **plot_kwargs):
        if columns is None:
            columns = ['val_loss', 'loss']

        history = pd.concat(
            [self.history(name, columns, folds) for name in names],
            axis=1
        )
        history.columns = [f'{name}_{col}' for name in names
                           for col in columns]

        if 'style' not in plot_kwargs:
            styles = ['-', '--', '-.', '.']
            plot_kwargs['style'] = styles[:len(columns)] * len(names)

        if 'color' not in plot_kwargs:
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                      'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                      'tab:olive', 'tab:cyan']
            plot_kwargs['color'] = [
                colors[i % len(colors)]
                for i in range(len(names))
                for _ in columns
            ]

        history.plot(**plot_kwargs)

    def times(self):
        return pd.DataFrame.from_dict(
            {name: self.results[name]['fit_time'].sum()
             for name in self.results},
            orient='index', columns=['time']
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


def all_columns_equal(array):
    """
    :param array: 2D numpy-array
    :return: True if all columns in input array are equal, False otherwise
    """
    first_column = array[:, 0].reshape((array.shape[0], 1))
    expected = first_column * np.ones((1, array.shape[1]))  # outer product
    return np.array_equal(array, expected)
