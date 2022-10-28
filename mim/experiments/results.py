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
    training_results: List[Result] = field(default_factory=list)
    validation_results: List[Result] = field(default_factory=list)

    def add(self, train_result: Result, validation_result: Result):
        self.training_results.append(train_result)
        self.validation_results.append(validation_result)

    @property
    def num_splits(self):
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
    def validation_targets(self):
        return pd.concat(
            [r.targets for r in self.validation_results], axis=0)

    @property
    def training_targets(self):
        return pd.concat(
            [r.targets for r in self.training_results], axis=0)

    @property
    def validation_predictions(self):
        return pd.concat(
            [r.predictions for r in self.validation_results],
            axis=0)

    @property
    def train_predictions(self):
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
            keys=range(self.num_splits),
            names=['split', 'epoch', 'target']
        )

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
            keys=range(self.num_splits),
            names=['split', 'epoch'],
            axis=1
        )

    def __str__(self):
        base, xp_name = self.path.split('/')[-2:]
        return f"Experiment result for {base}.{xp_name}"
