from dataclasses import dataclass, field
from typing import List, Union

import pandas as pd


@dataclass
class TrainingResult:
    fit_time: float = 0.0
    score_time: float = 0.0
    train_score: float = 0.0
    test_score: float = 0.0
    targets: pd.DataFrame = None
    predictions: pd.DataFrame = None
    history: dict = None
    model_summary: str = ''


@dataclass
class ExperimentResult:
    feature_names: Union[List, dict]
    metadata: dict
    experiment_summary: dict
    path: str = ''
    results: List[TrainingResult] = field(default_factory=list)

    def add(self, result: TrainingResult):
        self.results.append(result)

    @property
    def model_summary(self):
        return self.results[0].model_summary

    @property
    def num_splits(self):
        return len(self.results)

    @property
    def fit_times(self):
        return [r.fit_time for r in self.results]

    @property
    def score_times(self):
        return [r.score_time for r in self.results]

    @property
    def train_scores(self):
        return [r.train_score for r in self.results]

    @property
    def test_scores(self):
        return [r.test_score for r in self.results]

    @property
    def targets(self):
        return pd.concat([r.targets for r in self.results], axis=0)

    @property
    def predictions(self):
        return pd.concat([r.predictions for r in self.results], axis=0)

    @property
    def history(self):
        if self.results[0].history is None:
            return None

        return pd.concat(
            [pd.DataFrame(r.history) for r in self.results],
            keys=range(self.num_splits),
            names=['split', 'epoch'],
            axis=1
        )
