from dataclasses import dataclass, field
from typing import List, Union

import pandas as pd


@dataclass
class Result:
    time: float = 0.0
    score: float = 0.0
    targets: pd.DataFrame = None
    predictions: pd.DataFrame = None
    prediction_history: pd.DataFrame = None
    history: dict = field(default_factory=dict)


@dataclass
class TestResult:
    metadata: dict
    targets: pd.DataFrame
    predictions: pd.DataFrame


@dataclass
class ExperimentResult:
    feature_names: Union[List, dict]
    metadata: dict
    experiment_summary: dict
    path: str
    model_summary: str = ''
    total_splits: int = 1
    training_results: List[Result] = field(default_factory=list)
    validation_results: List[Result] = field(default_factory=list)
    test_results: Union[None, TestResult] = None

    def add(self, train_result: Result, validation_result: Result):
        self.training_results.append(train_result)
        self.validation_results.append(validation_result)

    @property
    def num_splits_done(self):
        return len(self.training_results)

    @property
    def training_times(self):
        return [r.time for r in self.training_results]

    @property
    def validation_times(self):
        return [r.time for r in self.validation_results]

    @property
    def training_scores(self):
        return [r.score for r in self.training_results]

    @property
    def validation_scores(self):
        return [r.score for r in self.validation_results]

    @property
    def test_targets(self):
        return self.test_results.targets

    @property
    def validation_targets(self):
        folds_total = len(self.validation_results)
        ensemble = self._ensemble()
        folds = folds_total // ensemble

        return pd.concat(
            [r.targets for r in self.validation_results[:folds]], axis=0)

    @property
    def training_targets(self):
        folds_total = len(self.training_results)
        ensemble = self._ensemble()
        folds = folds_total // ensemble

        return pd.concat(
            [r.targets for r in self.training_results[:folds]], axis=0)
        #
        # return pd.concat(
        #     [r.targets for r in self.training_results], axis=0)

    @property
    def test_predictions(self):
        return self.test_results.predictions

    @property
    def validation_predictions(self):
        # When using cross-validation I want to combine all predictions into
        # a single vector. When using ensembles, I want to average the
        # predictions across the ensembles. When using both, I first combine
        # all the cross-validation predictions, then I average across
        # ensembles. There are if-checks to avoid some unnecessary copying of
        # data (concat will copy data), but when using both ensembles and
        # cross-validation, there will be double-copying with this solution.
        # It's not the prettiest, but at least there's a few tests for it.
        def combine_cv_preds(start_id, stop_id):
            if stop_id - start_id > 1:
                return pd.concat(
                    [r.predictions for r in
                     self.validation_results[start_id:stop_id]],
                    axis=0
                )
            else:
                return self.validation_results[start_id].predictions

        total_splits = len(self.validation_results)
        ensemble_splits = self._ensemble()
        cv_splits = total_splits // ensemble_splits

        ensemble_preds = [
            combine_cv_preds(cv_splits * i, cv_splits * (i + 1))
            for i in range(ensemble_splits)
        ]
        if len(ensemble_preds) > 1:
            return pd.DataFrame(pd.concat(ensemble_preds, axis=1).mean(axis=1))
        else:
            return ensemble_preds[0]

    @property
    def training_predictions(self):
        return pd.concat(
            [r.predictions for r in self.training_results], axis=0)

    @property
    def training_prediction_history(self):
        return self._prediction_history(self.training_results)

    @property
    def validation_prediction_history(self):
        return self._prediction_history(self.validation_results)

    def _prediction_history(self, results):
        return pd.concat(
            [r.prediction_history for r in results],
            axis=1,
            keys=range(self.num_splits_done),
            names=['split', 'epoch', 'target']
        )

    def _ensemble(self):
        if 'ensemble' in self.metadata:
            return self.metadata['ensemble']
        else:
            return 1

    def _is_ensemble(self):
        return 'ensemble' in self.metadata and self.metadata['ensemble'] > 1

    @property
    def training_history(self):
        return self._result_history(self.training_results)

    @property
    def validation_history(self):
        return self._result_history(self.validation_results)

    def _result_history(self, results):
        if results[0].history is None:
            return None

        return pd.concat(
            [pd.DataFrame(r.history) for r in results],
            keys=range(self.num_splits_done),
            names=['split', 'epoch'],
            axis=1
        )

    def __str__(self):
        project, base, xp_name = self.path.split('/')[-3:]
        return (
            f"Experiment result for {project}.{base}.{xp_name}\n"
            f"Description: {self.experiment_summary['description']}\n"
            f"Completed: {self.metadata['timestamp']}\n"
            f"Commit: {self.metadata['current_commit']}"
        )

    def __repr__(self):
        return str(self)
