import pandas as pd

from mim.experiments.results import TrainResult, ExperimentResult


class TestExperimentResult:
    def test_predictions_from_single_fold(self):
        p = TrainResult(
            fit_time=0.1,
            score_time=0.2,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame([0.8, 0.2, 0.1, 0.6]),
        )
        r = ExperimentResult(
            results=[p],
            feature_names=['a', 'b'],
            metadata={},
            experiment_summary={},
        )

        assert r.validation_predictions.equals(p.validation_predictions)

    def test_predictions_from_multiple_folds(self):
        p1 = TrainResult(
            fit_time=0.1,
            score_time=0.2,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame([0.8, 0.2, 0.1, 0.6]),
        )
        p2 = TrainResult(
            fit_time=0.1,
            score_time=0.2,
            targets=pd.DataFrame([1, 1, 0, 0]),
            predictions=pd.DataFrame([0.8, 0.4, 0.1, 0.4]),
        )
        p3 = TrainResult(
            fit_time=0.1,
            score_time=0.2,
            targets=pd.DataFrame([0, 0, 0, 1, 1, 0, 0]),
            predictions=pd.DataFrame([0.1, 0.9, 0.2, 0.8, 0.4, 0.1, 0.4]),
        )
        r = ExperimentResult(
            results=[p1, p2, p3],
            feature_names=['a', 'b'],
            metadata={},
            experiment_summary={},
        )

        expected_predictions = pd.DataFrame(
            [0.8, 0.2, 0.1, 0.6, 0.8, 0.4, 0.1, 0.4, 0.1, 0.9, 0.2, 0.8,
             0.4, 0.1, 0.4],
            index=[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6]
        )
        print(r.validation_predictions)
        assert r.validation_predictions.equals(expected_predictions)

    def test_single_split_history(self):
        p1 = TrainResult(
            fit_time=0.1,
            score_time=0.2,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame([0.8, 0.2, 0.1, 0.6]),
            history={
                'loss': [1, 0.9, 0.7, 0.5],
                'val_loss': [3, 1.2, 0.9, 0.6]
            }
        )
        r = ExperimentResult(
            results=[p1],
            feature_names=['a', 'b'],
            metadata={},
            experiment_summary={},
        )
        expected_history = pd.DataFrame(
            [[1, 3], [0.9, 1.2], [0.7, 0.9], [0.5, 0.6]],
            columns=pd.MultiIndex.from_product(([0], ['loss', 'val_loss']))
        )
        assert r.history.equals(expected_history)

    def test_double_split_history(self):
        p1 = TrainResult(
            fit_time=0.1,
            score_time=0.2,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame([0.8, 0.2, 0.1, 0.6]),
            history={
                'loss': [1, 0.9, 0.7, 0.5],
                'val_loss': [3, 1.2, 0.9, 0.6]
            }
        )
        p2 = TrainResult(
            fit_time=0.1,
            score_time=0.2,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame([0.8, 0.2, 0.1, 0.6]),
            history={
                'loss': [1.1, 0.9, 0.8, 0.7],
                'val_loss': [3, 2, 1, 0.9]
            }
        )
        r = ExperimentResult(
            results=[p1, p2],
            feature_names=['a', 'b'],
            metadata={},
            experiment_summary={},
        )
        expected_history = pd.DataFrame(
            [[1, 3, 1.1, 3],
             [0.9, 1.2, 0.9, 2],
             [0.7, 0.9, 0.8, 1],
             [0.5, 0.6, 0.7, 0.9]],
            columns=pd.MultiIndex.from_product([[0, 1], ['loss', 'val_loss']])
        )
        assert r.history.equals(expected_history)
