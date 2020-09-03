import pandas as pd

from mim.cross_validation import KFold
from mim.fakes.generate_fake_data import make_fake_multi_index


class TestKFoldCrossValidation:
    """Note that this K-Fold cross validation is using randomness in order to
    select the sets. Set a random seed if that is an issue for future tests."""

    def test_run_cross_validation(self):
        n_splits = 3
        cv = KFold(n_splits, level='PatientID')
        multi_index = make_fake_multi_index(
            pids=list(range(3)),
            start='2015-01-01',
            end='2015-06-30'
        )

        for _, _ in cv.split(multi_index):
            pass

    def test_assert_correct_sizes_and_iterations_for_pid(self):
        n_splits = 3
        cv = KFold(n_splits, level='PatientID')

        all_patients = list(range(3))
        multi_index = make_fake_multi_index(
            pids=all_patients,
            start='2015-01-01',
            end='2015-06-30'
        )

        index_df = pd.DataFrame(index=multi_index, columns=['index'])
        index_df['index'] = range(len(index_df))

        iterations = 0
        for train, test in cv.split(multi_index):
            iterations += 1

            train_df = index_df.iloc[train]
            test_df = index_df.iloc[test]

            patients_in_train = set(
                train_df.index.get_level_values('PatientID'))

            patients_in_test = set(
                test_df.index.get_level_values('PatientID'))

            assert patients_in_test.isdisjoint(patients_in_train)
            assert set(all_patients) == patients_in_train | patients_in_test

        assert iterations == n_splits

    def test_assert_correct_sizes_and_iterations_for_time(self):
        n_splits = 3
        cv = KFold(n_splits, level='Time')

        all_patients = list(range(3))
        multi_index = make_fake_multi_index(
            pids=all_patients,
            start='2015-01-01',
            end='2015-06-30'
        )

        index_df = pd.DataFrame(index=multi_index, columns=['index'])
        index_df['index'] = range(len(index_df))
        all_times = set(multi_index.get_level_values('Time'))

        iterations = 0
        for train, test in cv.split(multi_index):
            iterations += 1

            train_df = index_df.iloc[train]
            test_df = index_df.iloc[test]

            times_in_train = set(
                train_df.index.get_level_values('Time'))

            times_in_test = set(
                test_df.index.get_level_values('Time'))

            assert times_in_test.isdisjoint(times_in_train)
            assert all_times == times_in_train | times_in_test

        assert iterations == n_splits
