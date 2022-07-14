import pandas as pd

from mim.experiments.results import Result, ExperimentResult


class TestExperimentResult:
    def test_predictions_from_single_fold(self):
        p = Result(
            time=0.1,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame([0.8, 0.2, 0.1, 0.6]),
        )
        r = ExperimentResult(
            validation_results=[p],
            feature_names=['a', 'b'],
            metadata={},
            path='',
            experiment_summary={},
        )

        assert r.validation_predictions.equals(p.predictions)

    def test_predictions_from_multiple_folds(self):
        p1 = Result(
            time=0.1,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame([0.8, 0.2, 0.1, 0.6]),
        )
        p2 = Result(
            time=0.1,
            targets=pd.DataFrame([1, 1, 0, 0]),
            predictions=pd.DataFrame([0.8, 0.4, 0.1, 0.4]),
        )
        p3 = Result(
            time=0.1,
            targets=pd.DataFrame([0, 0, 0, 1, 1, 0, 0]),
            predictions=pd.DataFrame([0.1, 0.9, 0.2, 0.8, 0.4, 0.1, 0.4]),
        )
        r = ExperimentResult(
            validation_results=[p1, p2, p3],
            feature_names=['a', 'b'],
            metadata={},
            path='',
            experiment_summary={},
        )

        expected_predictions = pd.DataFrame(
            [0.8, 0.2, 0.1, 0.6, 0.8, 0.4, 0.1, 0.4, 0.1, 0.9, 0.2, 0.8,
             0.4, 0.1, 0.4],
            index=[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6]
        )
        print(r.validation_predictions)
        assert r.validation_predictions.equals(expected_predictions)

    def test_single_split_validation_history(self):
        history = pd.DataFrame(
            [[0.5, 0.6, 0.7, 0.8, 0.9],
             [0.5, 0.4, 0.3, 0.2, 0.1],
             [0.5, 0.2, 0.1, 0.0, 0.0],
             [0.5, 0.6, 0.9, 0.7, 0.8]],
            columns=pd.MultiIndex.from_product(
                (range(5), ['foo']), names=['epoch', 'target']
            )
            # columns=pd.Index(range(5), name='epoch')
        )
        p1 = Result(
            time=0.1,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame([0.8, 0.2, 0.1, 0.6]),
            prediction_history=history
        )
        r = ExperimentResult(
            training_results=[p1],
            validation_results=[p1],
            feature_names=['a', 'b'],
            metadata={},
            path='',
            experiment_summary={},
        )
        expected_history = pd.DataFrame(
            [[0.5, 0.6, 0.7, 0.8, 0.9],
             [0.5, 0.4, 0.3, 0.2, 0.1],
             [0.5, 0.2, 0.1, 0.0, 0.0],
             [0.5, 0.6, 0.9, 0.7, 0.8]],
            columns=pd.MultiIndex.from_product(
                ([0], range(5), ['foo']),
                names=['split', 'epoch', 'target']
            )
        )
        assert r.validation_prediction_history.equals(expected_history)

    def test_double_split_validation_history(self):
        h1 = pd.DataFrame(
            [[0.5, 0.6, 0.7, 0.8, 0.9],
             [0.5, 0.4, 0.3, 0.2, 0.1],
             [0.5, 0.2, 0.1, 0.0, 0.0],
             [0.5, 0.6, 0.9, 0.7, 0.8]],
            columns=pd.MultiIndex.from_product(
                (range(5), ['foo']), names=['epoch', 'target']
            )
        )
        h2 = pd.DataFrame(
            [[0.5, 0.6, 0.8, 0.9, 0.9],
             [0.5, 0.3, 0.5, 0.4, 0.3],
             [0.5, 0.5, 0.4, 0.3, 0.0],
             [0.5, 0.6, 0.9, 0.8, 0.7]],
            columns=pd.MultiIndex.from_product(
                (range(5), ['foo']), names=['epoch', 'target']
            )
        )
        p1 = Result(
            time=0.2,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame([0.8, 0.2, 0.1, 0.6]),
            prediction_history=h1
        )
        p2 = Result(
            time=0.2,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame([0.8, 0.2, 0.1, 0.6]),
            prediction_history=h2
        )
        r = ExperimentResult(
            training_results=[p1, p2],
            validation_results=[p1, p2],
            feature_names=['a', 'b'],
            metadata={},
            path="",
            experiment_summary={},
        )
        expected_history = pd.DataFrame(
            [[0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.6, 0.8, 0.9, 0.9],
             [0.5, 0.4, 0.3, 0.2, 0.1, 0.5, 0.3, 0.5, 0.4, 0.3],
             [0.5, 0.2, 0.1, 0.0, 0.0, 0.5, 0.5, 0.4, 0.3, 0.0],
             [0.5, 0.6, 0.9, 0.7, 0.8, 0.5, 0.6, 0.9, 0.8, 0.7]],
            columns=pd.MultiIndex.from_product(
                ([0, 1], range(5), ['foo']),
                names=['split', 'epoch', 'target']
            )
        )
        assert r.validation_prediction_history.equals(expected_history)
