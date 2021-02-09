from unittest import mock

import pandas as pd
import pytest
from freezegun import freeze_time

from mim.util.metadata import Metadata, Validator, MetadataConsistencyError


class TestMetadata:
    expected_files = {
        './data/foo/bar.pickle':
            {'changed': '2018-07-11 13:14:42'},
        './data/foo/baz.csv':
            {'changed': '2018-07-26 10:14:54'},
        './data/spam/eggs_and_bacon.csv':
            {'changed': '2018-07-26 10:14:58'},
        './data/spam/ham.csv':
            {'changed': '2018-07-26 10:16:06'},
        './data/secret/sauce.csv':
            {'changed': '2018-07-26 10:16:13'}}

    @freeze_time('2017-05-21 12:00:05')
    def test_can_get_current_time(self):
        md = Metadata()
        assert '2017-05-21 12:00:05' == md.timestamp()

    def test_current_time_is_not_2017(self):
        md = Metadata()
        assert '2017-05-21 12:00:05' != md.timestamp()

    @freeze_time('2021-05-22 08:04:15')
    @mock.patch('mim.util.metadata.Metadata.current_commit')
    @mock.patch('mim.util.metadata.Metadata.has_uncommitted_changes')
    @mock.patch('mim.util.metadata.Metadata.current_branch')
    @mock.patch('mim.util.metadata.Metadata.file_data')
    def test_metadata_report(self, mock_file_data, mock_branch,
                             mock_uncommitted, mock_current):
        mock_file_data.return_value = self.expected_files
        mock_branch.return_value = 'feature-important_improvement-AI-999'
        mock_uncommitted.return_value = False
        mock_current.return_value = 'abcdefgh'

        expected = {
            'current_branch': 'feature-important_improvement-AI-999',
            'has_uncommitted_changes': False,
            'current_commit': 'abcdefgh',
            'file_data': self.expected_files,
            'timestamp': '2021-05-22 08:04:15'}

        md = Metadata()

        assert expected == md.report(conda=False)

        include = {}
        while expected:
            k, v = expected.popitem()
            include[k] = False
            assert expected == md.report(conda=False, **include)


class TestValidator:
    def test_identical_metadata_is_valid(self):
        metadata = {'a': 1, 'b': 2}
        v = Validator()
        assert v.validate_consistency([metadata, metadata])

    def test_can_validate_uncommitted_changes(self):
        metadata = {'has_uncommitted_changes': False}
        v = Validator(allow_uncommitted=False)
        assert v.validate_consistency([metadata])

        with pytest.raises(MetadataConsistencyError):
            metadata = {'has_uncommitted_changes': True}
            v.validate_consistency([metadata])

    def test_missing_uncommitted_changes_metadata_raises_error(self):
        with pytest.raises(MetadataConsistencyError):
            v = Validator(allow_uncommitted=False)
            v.validate_consistency([{}])

    def test_parameters_respected(self):
        v = Validator(allow_uncommitted=True)
        assert v.validate_consistency([{}])
        assert v.validate_consistency([{'has_uncommitted_changes': True}])
        assert v.validate_consistency([{'has_uncommitted_changes': False}])

    def test_some_missing_uncommitted_changes_metadata_raises_error(self):
        with pytest.raises(MetadataConsistencyError):
            v = Validator(allow_uncommitted=False)
            v.validate_consistency([{'has_uncommitted_changes': True},
                                    {'has_uncommitted_changes': False},
                                    {}])

    def test_same_commit_is_valid(self):
        v = Validator(allow_different_commits=False)
        assert v.validate_consistency([
            {'current_commit': 'abc'},
            {'current_commit': 'abc'},
            {'current_commit': 'abc', 'has_uncommitted_changes': True}])

    def test_different_commits_raises_error(self):
        v = Validator(allow_different_commits=False)
        with pytest.raises(MetadataConsistencyError):
            v.validate_consistency([
                {'current_commit': 'abcd'},
                {'current_commit': 'efgh'}])

        with pytest.raises(MetadataConsistencyError):
            v.validate_consistency([
                {'current_commit': 'abcd'},
                {'current_commit': 'abcd'},
                {'current_commit': 'abcde'}])

    def test_timestamp_difference_within_bounds(self):
        v = Validator(max_age_difference=pd.Timedelta(days=10))
        assert v.validate_consistency([
            {'timestamp': '2017-01-01'},
            {'timestamp': '2017-01-01'}
        ])

        assert v.validate_consistency([
            {'timestamp': '2017-01-01'},
            {'timestamp': '2017-01-01'},
            {'timestamp': '2017-01-10 12:00:01'}
        ])

    def test_timestamp_difference_out_of_bounds_raises_error(self):
        v = Validator(max_age_difference=pd.Timedelta(days=10))
        with pytest.raises(MetadataConsistencyError):
            v.validate_consistency([
                {'timestamp': '2017-01-01'},
                {'timestamp': '2017-01-21'}
            ])

        with pytest.raises(MetadataConsistencyError):
            v.validate_consistency([
                {'timestamp': '2017-01-01'},
                {'timestamp': '2017-01-02 12:00:00'},
                {'timestamp': '2017-01-21'}
            ])

    def test_same_file_data_is_consistent(self):
        v = Validator(allow_different_files=False)
        assert v.validate_consistency([
            {'file_data': {
                'patient_list.csv': {'changed': '2017-01-01 10:00:00'},
                'patient_height.csv': {'changed': '2017-01-01 10:00:00'}
            }},
            {'file_data': {
                'patient_list.csv': {'changed': '2017-01-01 10:00:00'},
                'patient_height.csv': {'changed': '2017-01-01 10:00:00'}
            }}])

        assert v.validate_consistency([
            {'file_data': {
                'patient_list.csv': {'changed': '2017-01-01 10:00:00'},
                'patient_height.csv': {'changed': '2017-01-01 10:00:00'}
            }},
            {'file_data': {
                'patient_height.csv': {'changed': '2017-01-01 10:00:00'},
                'patient_list.csv': {'changed': '2017-01-01 10:00:00'}
            }}])

    def test_different_file_data_raises_error(self):
        v = Validator(allow_different_files=False)
        with pytest.raises(MetadataConsistencyError):
            v.validate_consistency([
                {'file_data': {
                    'patient_list.csv': {'changed': '2017-01-01 10:00:00'},
                    'patient_height.csv': {'changed': '2017-01-01 10:00:00'}
                }},
                {'file_data': {
                    'patient_list.csv': {'changed': '2017-01-01 10:00:00'},
                    'patient_height.csv': {'changed': '2017-01-01 10:00:01'}
                }}])

        with pytest.raises(MetadataConsistencyError):
            v.validate_consistency([
                {'file_data': {
                    'patient_list.csv': {'changed': '2017-01-01 10:00:00'},
                    'patient_height.csv': {'changed': '2017-01-02 10:02:00'}
                }},
                {'file_data': {
                    'patient_list.csv': {'changed': '2017-01-01 10:00:00'},
                    'patient_height.csv': {'changed': '2017-01-02 10:02:00'},
                    'patient_height2.csv': {'changed': '2017-01-01 10:00:00'}
                }}])

        with pytest.raises(MetadataConsistencyError):
            v.validate_consistency([
                {'file_data': {}},
                {'file_data': {
                    'patient_list.csv': {'changed': '2017-01-01 10:00:00'},
                }}])

        with pytest.raises(MetadataConsistencyError):
            v.validate_consistency([
                {'file_data': {
                    'patient_list.csv': {'changed': '2017-01-01 10:00:00'},
                    'patient_height.csv': {'changed': '2017-01-02 10:02:00'}
                }},
                {'file_data': {
                    'patient_list.csv': {'changed': '2017-01-01 10:00:00'},
                    'patientheight.csv': {'changed': '2017-01-02 10:02:00'},
                }}])

        with pytest.raises(MetadataConsistencyError):
            v.validate_consistency([
                {'file_data': {
                    'patient_list.csv': {'changed': '2017-01-01 10:00:00'},
                    'patient_height.csv': {'changed': '2017-01-02 10:02:00'}
                }},
                {'data_file': {
                    'patient_list.csv': {'changed': '2017-01-01 10:00:00'},
                    'patient_height.csv': {'changed': '2017-01-02 10:02:00'},
                }}])
