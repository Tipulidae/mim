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
        expected = p.predictions
        actual = r.validation_predictions
        print(f"{expected=}")
        print(f"{actual=}")
        pd.testing.assert_frame_equal(expected, actual)
        # assert r.validation_predictions.equals()

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

    def test_ensemble_targets_single_fold(self):
        p1 = Result(
            time=0.1,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame([0.8, 0.2, 0.1, 0.6]),
        )
        p2 = Result(
            time=0.1,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame([0.7, 0.3, 0.1, 0.5]),
        )
        r = ExperimentResult(
            validation_results=[p1, p2],
            feature_names=['a', 'b'],
            metadata={'ensemble': 2},
            path='',
            experiment_summary={},
        )
        assert r.validation_targets.equals(p1.targets)

    def test_ensemble_targets_multiple_folds(self):
        p1 = Result(
            time=0.1,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame([0.8, 0.2, 0.1, 0.6]),
        )
        p2 = Result(
            time=0.1,
            targets=pd.DataFrame([1, 1, 0, 0]),
            predictions=pd.DataFrame([0.7, 0.9, 0.1, 0.2]),
        )
        p3 = Result(
            time=0.1,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame([0.6, 0.3, 0.2, 0.9]),
        )
        p4 = Result(
            time=0.1,
            targets=pd.DataFrame([1, 1, 0, 0]),
            predictions=pd.DataFrame([0.8, 0.8, 0.2, 0.1]),
        )
        r = ExperimentResult(
            validation_results=[p1, p2, p3, p4],
            feature_names=['a', 'b'],
            metadata={'ensemble': 2},
            path='',
            experiment_summary={},
        )
        expected = pd.concat([p1.targets, p2.targets], axis=0)
        assert r.validation_targets.equals(expected)

    def test_ensemble_predictions_single_fold(self):
        p1 = Result(
            time=0.1,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame([0.8, 0.2, 0.1, 0.6]),
        )
        p2 = Result(
            time=0.1,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame([0.6, 0.4, 0.1, 0.4]),
        )
        r = ExperimentResult(
            validation_results=[p1, p2],
            feature_names=['a', 'b'],
            metadata={'ensemble': 2},
            path='',
            experiment_summary={},
        )
        expected = pd.DataFrame(
            [0.7, 0.3, 0.1, 0.5],

            # columns=pd.Index([0], name='fold'),
        )
        actual = r.validation_predictions
        print(f"{expected=}")
        print(f"{actual=}")
        # print(f"{actual.columns=}")
        pd.testing.assert_frame_equal(expected, actual)

    def test_ensemble_predictions_multiple_folds(self):
        p1 = Result(
            time=0.1,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame(
                [0.8, 0.2, 0.1, 0.6],
                index=pd.Index(range(4), name='Alias')
            ),
        )
        p2 = Result(
            time=0.1,
            targets=pd.DataFrame([1, 1, 0, 0]),
            predictions=pd.DataFrame(
                [0.7, 0.9, 0.1, 0.2],
                index=pd.Index(range(4, 8), name='Alias')
            ),
        )
        p3 = Result(
            time=0.1,
            targets=pd.DataFrame([1, 0, 0, 1]),
            predictions=pd.DataFrame(
                [0.6, 0.4, 0.3, 0.8],
                index=pd.Index(range(4), name='Alias')
            ),
        )
        p4 = Result(
            time=0.1,
            targets=pd.DataFrame([1, 1, 0, 0]),
            predictions=pd.DataFrame(
                [0.7, 0.7, 0.3, 0.0],
                index=pd.Index(range(4, 8), name='Alias')
            ),
        )
        r = ExperimentResult(
            validation_results=[p1, p2, p3, p4],
            feature_names=['a', 'b'],
            metadata={'ensemble': 2},
            path='',
            experiment_summary={},
        )
        expected = pd.DataFrame(
            [0.7, 0.3, 0.2, 0.7, 0.7, 0.8, 0.2, 0.1],
            index=pd.Index(range(8), name='Alias')
        )
        actual = r.validation_predictions
        print(f"{expected=}")
        print(f"{actual=}")
        pd.testing.assert_frame_equal(expected, r.validation_predictions)

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
