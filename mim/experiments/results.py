from dataclasses import dataclass, field
from typing import List, Union

import pandas as pd


@dataclass
class Result:
    time: float = 0.0
    score: float = 0.0
    targets: pd.DataFrame = None
    predictions: pd.DataFrame = None
    history: pd.DataFrame = None


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
    def train_times(self):
        return [r.time for r in self.training_results]

    @property
    def validation_times(self):
        return [r.time for r in self.validation_results]

    @property
    def train_scores(self):
        return [r.score for r in self.training_results]

    @property
    def validation_scores(self):
        return [r.score for r in self.validation_results]

    @property
    def validation_targets(self):
        return pd.concat(
            [r.targets for r in self.validation_results], axis=0)

    @property
    def train_targets(self):
        return pd.concat(
            [r.targets for r in self.training_results], axis=0)

    @property
    def validation_predictions(self):
        return pd.concat(
            [r.predictions for r in self.validation_results], axis=0)

    @property
    def train_predictions(self):
        return pd.concat(
            [r.predictions for r in self.training_results], axis=0)

    @property
    def training_history(self):
        return pd.concat(
            [r.history for r in self.training_results],
            axis=1,
            keys=range(self.num_splits),
            names=['split', 'epoch']
        )

    @property
    def validation_history(self):
        return pd.concat(
            [r.history for r in self.validation_results],
            axis=1,
            keys=range(self.num_splits),
            names=['split', 'epoch']
        )

    @property
    def history(self):
        if self.training_results[0].history is None:
            return None

        return pd.concat(
            [pd.DataFrame(r.history) for r in self.training_results],
            keys=range(self.num_splits),
            names=['split', 'epoch'],
            axis=1
        )
