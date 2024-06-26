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
    roc_curve,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from mim.util.logs import get_logger
from mim.util.util import ranksort, insensitive_iglob, is_categorical
from mim.util.metrics import (
    positive_predictive_value,
    negative_predictive_value,
    rule_in_rule_out
)
from mim.experiments.hyper_parameter import flatten
from mim.config import PATH_TO_TEST_RESULTS

log = get_logger("Presenter")

classification_scores = {
    'auc': roc_auc_score,
}
classification_threshold_scores = {
    'f1': f1_score,
    'accuracy': accuracy_score,
    'recall': recall_score,
    'precision': precision_score,
    'ppv': positive_predictive_value,
    'npv': negative_predictive_value
}
regression_scores = {
    'r2': r2_score,
    'mae': mean_absolute_error,
    'mse': mean_squared_error
}


class Results:
    def __init__(self, name, verbose=2):
        self.results = dict()

        dev_paths = insensitive_iglob(
            f"{PATH_TO_TEST_RESULTS}/{name}/**/train_val_results.pickle",
            recursive=True
        )
        for path in sorted(dev_paths):
            _, xp_name = os.path.split(os.path.dirname(path))
            if verbose > 1:
                log.info(f"Loading {xp_name}")
            if xp_name in self.results:
                log.warning(f"Two experiments with the name {xp_name}!")

            self.results[xp_name] = pd.read_pickle(path)

            test_path = os.path.join(
                os.path.dirname(path), 'test_results.pickle')
            if os.path.isfile(test_path):
                self.results[xp_name].test_results = pd.read_pickle(test_path)

        if verbose > 0:
            log.info(f"Finished loading {len(self.results)} experiments.")

    def like(self, pattern):
        p = re.compile(pattern)
        for name in filter(p.match, self.results):
            yield self[name]

    def __len__(self):
        return len(self.results)

    def __getitem__(self, key):
        return self.results[key]

    def __contains__(self, key):
        return key in self.results


def scores(results, auc=True, rule_in_out=False):
    scores = []
    for result in results:
        targets = result.validation_targets.values.ravel()
        predictions = result.validation_predictions.values.ravel()

        data = []
        index = []
        if auc:
            data.append(roc_auc_score(targets, predictions))
            index.append('auc')
        if rule_in_out:
            riro = rule_in_rule_out(targets, predictions).mean(axis=0)
            data.extend(list(riro))
            index.extend(['rule-in', 'intermediate', 'rule-out'])

        s = pd.Series(data=data, index=index, name=result.name)
        scores.append(s)
    return pd.DataFrame(scores)


class Presenter:
    def __init__(self, name, verbose=2, legacy_path=False):
        self.results = dict()
        if legacy_path:
            result_name = 'results'
        else:
            result_name = 'train_val_results'
        paths = insensitive_iglob(
            f"{PATH_TO_TEST_RESULTS}/{name}/**/{result_name}.pickle",
            recursive=True
        )

        for path in sorted(paths):
            _, xp_name = os.path.split(os.path.dirname(path))
            if verbose > 1:
                log.info(f"Loading {xp_name}")
            if xp_name in self.results:
                log.warning(f"Two experiments with the name {xp_name}!")

            # trunk-ignore(bandit/B301)
            self.results[xp_name] = pd.read_pickle(path)

        if verbose > 0:
            log.info(f"Finished loading {len(self.results)} experiments.")

    def describe(self, like='.*'):
        results = []
        for name, xp in self._results_that_match_pattern(like):
            results.append(pd.Series(
                data=[
                    np.mean(xp.validation_scores),
                    np.std(xp.validation_scores),
                    np.mean(xp.training_scores),
                    xp.metadata['current_commit'][:8],
                    xp.metadata['has_uncommitted_changes'],
                    xp.metadata['timestamp']],
                index=[
                    'test_score',
                    'test_score_std',
                    'train_score',
                    'commit',
                    'changed',
                    'timestamp'],
                name=name))
        return pd.DataFrame(results)

    def summary(self, like='.*', include_scores=True):
        xps = self._results_that_match_pattern(like)

        flat_results = []
        for name, xp in xps:
            if 'experiment_summary' in xp:
                flat = flatten(xp['experiment_summary'])
                flat_results.append(pd.Series(flat, name=name))

        df = pd.concat(flat_results, axis=1).T
        if include_scores:
            return df.join(self.scores()).T
        else:
            return df.T

    def train_test_scores(self, name):
        xp = self.results[name]
        return pd.DataFrame(
            [np.array(xp['train_score']), xp['test_score']],
            index=['train', 'test']
        )

    def scores(self, like='.*', auc=True, rule_in_out=False, legacy=False):
        results = []
        for name, xp in list(self._results_that_match_pattern(like)):
            if legacy:
                targets, predictions = self._target_predictions(name)
                targets = targets.values.ravel()
                predictions = predictions.values.ravel()
            else:
                targets = xp.validation_targets.values.ravel()
                predictions = xp.validation_predictions.values.ravel()

            data = []
            index = []
            if auc:
                data.append(roc_auc_score(targets, predictions))
                index.append('auc')
            if rule_in_out:
                riro = rule_in_rule_out(targets, predictions).mean(axis=0)
                data.extend(list(riro))
                index.extend(['rule-in', 'intermediate', 'rule-out'])

            s = pd.Series(data=data, index=index, name=name)
            results.append(s)
        return pd.DataFrame(results)

    def scores2(self, like='.*', threshold=None):
        results = {}
        for name, _ in list(self._results_that_match_pattern(like)):
            targets, predictions = self._target_predictions(name)
            results[name] = pd.DataFrame({
                col: calculate_scores(targets[col], predictions[col],
                                      threshold=threshold)
                for col in targets.columns
            })

        return pd.concat(results.values(), axis=1, keys=results.keys())

    def predictions(self, like='.*', droplevel=True):
        # Return dataframe with the true targets and predictions for each
        # experiment
        predictions = []
        keys = []
        the_target = None
        for name, xp in self._results_that_match_pattern(like):
            target = xp.validation_targets
            prediction = xp.validation_predictions
            if the_target is None:
                the_target = target

            if not the_target.equals(target):
                raise Exception('Targets differ')
            predictions.append(prediction)
            keys.append(name)

        result = pd.concat(
            [the_target] + predictions,
            keys=['target'] + keys,
            axis=1
        )
        if droplevel:
            result.columns = result.columns.droplevel(1)
        return result

    def prediction_ranks(self, like='.*'):
        predictions = self.predictions(like=like)
        for col in predictions.columns[1:]:
            predictions.loc[:, col] = ranksort(predictions.loc[:, col])

        predictions.iloc[:, 1:] /= len(predictions)
        return predictions

    def threshold_scores(self, like='.*', threshold=0.5):
        results = []
        for name, _ in self._results_that_match_pattern(like):
            targets, predictions = self._threshold_target_predictions(
                name, threshold)
            results.append(pd.Series(
                data=[
                    precision_score(targets, predictions),
                    recall_score(targets, predictions),
                    accuracy_score(targets, predictions),
                    f1_score(targets, predictions),
                    positive_predictive_value(targets, predictions),
                    negative_predictive_value(targets, predictions)
                ],
                index=[
                    'precision',
                    'recall',
                    'accuracy',
                    'f1',
                    'ppv',
                    'npv'
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
            k, = plt.plot(fpr, tpr, lw=1, alpha=1)
            lines.append(k)
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

        # plt.gca().add_patch(
        #     patches.Rectangle(
        #         (0, 0), 0.1, 1.0, linewidth=0, alpha=0.1, facecolor='red')
        # )
        # plt.gca().add_patch(
        #     patches.Rectangle(
        #         (0, 0.99), 1.0, 0.01, linewidth=0, alpha=0.1,
        #         facecolor='green')
        # )

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
            name, threshold
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

    def loss_history(self, name, avg=False):
        """
        :param name:
        :param avg: Whether or not to average the loss across the folds
        :return:
        """
        xp = self.results[name]
        loss = xp.training_history.loc[:, pd.IndexSlice[:, 'loss']]
        if avg:
            return loss.mean(axis=1)
        else:
            return loss

    def history(self, name, columns='all', folds='all'):
        xp = self.results[name]
        return xp.validation_history
        # if xp['history'] is None:
        #     print(f"Experiment {name} has no history.")
        #     return
        #
        # history = pd.concat(
        #     [pd.DataFrame(h) for h in xp['history']],
        #     axis=1,
        #     keys=[f'fold {i}' for i in range(len(xp['history']))]
        # )
        #
        # if isinstance(columns, list):
        #     history = history.loc[:, pd.IndexSlice[:, columns]]
        # if isinstance(folds, list):
        #     fold_names = [f'fold {i}' for i in folds]
        #     history = history.loc[:, pd.IndexSlice[fold_names, :]]
        # if folds == 'first':
        #     history = history.loc[:, 'fold 0']
        #
        # return history

    def max_auc(self, like='.*', column='val_auc'):
        def get_max_auc(xp):
            return xp.validation_history[0][column].max()

        results = {}
        for name, xp in self._results_that_match_pattern(like):
            results[name] = get_max_auc(xp)

        return pd.Series(results)

    def best_during_training(self, column='auc', bigger_is_better=True):
        def best_and_final(xp):
            history = self.history(
                xp, columns=[f'val_{column}'], folds='first'
            )
            best = history.max()[0] if bigger_is_better else history.min()[0]
            final = history.iloc[-1][0]
            return best, final

        score = pd.DataFrame(
            ({
                f'final_{column}': final,
                f'best_{column}': best
            } for best, final in [best_and_final(xp) for xp in self.results]),
            index=self.results.keys()
        )
        score['overfit'] = score.iloc[:, 1] - score.iloc[:, 0]
        return score

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

    def last_history(self, like='.*'):
        results = []
        for name, xp in self._results_that_match_pattern(like):
            results.append(
                pd.Series(
                    {k: v[-1] for k, v in xp['history'][0].items()},
                    name=name
                )
            )

        return pd.DataFrame(results)

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

    def _threshold_target_predictions(self, name, threshold):
        targets, predictions = self._target_predictions(name)
        predictions = (predictions >= threshold).astype(int)
        return targets, predictions

    def _target_predictions(self, name):
        xp = self.results[name]
        targets = pd.concat(xp['targets'])
        predictions = pd.concat(xp['predictions']['prediction'], axis=0)
        predictions.index = targets.index
        predictions.columns = targets.columns
        return targets, predictions


def all_columns_equal(array):
    """
    :param array: 2D numpy-array
    :return: True if all columns in input array are equal, False otherwise
    """
    first_column = array[:, 0].reshape((array.shape[0], 1))
    expected = first_column * np.ones((1, array.shape[1]))  # outer product
    return np.array_equal(array, expected)


def calculate_scores(targets, predictions, threshold=None):
    if is_categorical(targets):
        return calculate_classification_scores(
            targets, predictions, threshold=threshold
        )
    else:
        return calculate_regression_scores(targets, predictions)


def calculate_classification_scores(targets, predictions, threshold=None):
    results = {}
    for name, scorer in classification_scores.items():
        results[name] = scorer(targets, predictions)

    if threshold:
        scores_th = (predictions >= threshold).astype(int)
        for name, scorer in classification_threshold_scores.items():
            results[name] = scorer(targets, scores_th)

    return results


def calculate_regression_scores(targets, predictions):
    results = {}
    for name, scorer in regression_scores.items():
        results[name] = scorer(targets, predictions)

    return results


def plot_calibration_curve(
        targets, predictions, bins=25, strategy='quantile',
        xlim=(-0.05, 1.05), ylim=(-0.05, 1.05)):
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    fraction_of_positives, mean_predicted_value = calibration_curve(
        targets, predictions, n_bins=bins, strategy=strategy)
    ax1.plot(mean_predicted_value, fraction_of_positives, "-",
             label='Predicted')
    ax2.hist(predictions, range=(0, 1), bins=bins, label='Prediction',
             histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim(ylim)
    ax1.set_xlim(xlim)
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()
