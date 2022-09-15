import subprocess
from os import listdir
from os.path import isfile, join, getmtime, abspath
from datetime import datetime

import pandas as pd


class Metadata:
    """
    This class is responsible for generating relevant metadata about the
    current state of the project. It can determine the current git branch,
    the current commit and whether there are uncommitted changes. It also
    takes note of the current time as well as the files in the specified
    folders and when they were last changed.

    Normal usage is to simply call Metadata().report(), which returns a
    dictionary containing the metadata. A list of reports can later be
    inspected and validated by the Validator class.
    """

    def __init__(self, folders=tuple()):
        self.metadata_folders = folders

    def report(self, as_string=False, **include):
        """
        Generates a report of the current metadata situation for the
        project, in the form of a dictionary. Contains current branch,
        current commit, whether there are uncommitted changes, timestamp
        and a list of data files and their last modified date.

        :param include: key-word-arguments corresponding to what should be
        included in the report. Default is to include everything,
        and individual report details can be turned, e.g.
        report(current_branch=False, has_uncommitted_changes=False).
        :return: Dictionary containing metadata.
        """
        include['report'] = False

        report_dict = {}
        for name, f in public_methods(self):
            if name not in include or include[name]:
                report_dict[name] = f()

        if as_string:
            return dict_to_string(report_dict)
        else:
            return report_dict

    def current_branch(self):
        """
        Returns the name of the current branch. Equivalent to the git command
        $ git rev-parse --abbrev-ref HEAD
        :return: Name of current branch.
        """
        return self._bash_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])

    def has_uncommitted_changes(self):
        """
        Whether the current branch contains uncommitted changes.
        :return: True if there are uncommitted changes, False otherwise.
        """
        return len(self._bash_command(
            ['git', 'diff', 'HEAD', '--name-only'])) != 0

    def current_commit(self):
        """
        Returns the hash of the current commit, equivalent to the git command
        git rev-parse HEAD
        :return: Hash of current commit.
        """
        return self._bash_command(['git', 'rev-parse', 'HEAD'])

    @staticmethod
    def timestamp():
        """
        Returns the current time, in ISO standard: YYYY-MM-DD HH:MM:SS.ms
        :return: Current time, as a string.
        """
        return str(datetime.now())

    def file_data(self):
        """
        Returns a dictionary where the keys are the names of the files in the
        data/csv and data/cdc folders, and the values are dictionaries. The
        value dictionaries contains the key 'changed', where the value is
        the date that the file was last modified.
        :return: Dictionary of file names and the date they were last
        changed.
        """
        return {f: {'changed': self._file_modified_date(f)} for f in
                self._all_files()}

    def conda(self):
        """
        :return: Returns the output from running "conda list", which is to
        say, all the installed packages in the currently active environment.
        """
        return self._bash_command(['conda', 'list'])

    @staticmethod
    def _bash_command(array):
        return subprocess.check_output([] + array).strip().decode('UTF8')

    def _all_files(self):
        for path in self.metadata_folders:
            for name in listdir(path):
                full_name = join(path, name)

                if isfile(full_name):
                    yield abspath(full_name)

    @staticmethod
    def _file_modified_date(file_name):
        changed_timestamp = getmtime(file_name)
        return datetime.fromtimestamp(changed_timestamp).\
            strftime('%Y-%m-%d %H:%M:%S')


class Validator:
    def __init__(self, allow_uncommitted=True, allow_different_commits=True,
                 allow_different_branches=True, allow_different_files=True,
                 allow_different_environments=True,
                 max_age_difference=None):
        """
        Each Experiment in mim is associated with Metadata information. The
        purpose of the Validator class is to consider this metadata from
        one or several experiments and determine whether the metadata is
        consistent. Exactly what it means for a collection of metadata
        reports to be consistent may depend on different things, which are
        specified in the constructor.

        Once created, the validate_consistency method can be called with a
        list of metadata reports (dictionaries). This will return True if
        the reports are consistent, otherwise it will raise a
        MetadataConsistencyError.

        :param allow_uncommitted: If False, none of the metadata reports
        must indicate uncommitted changes to be considered consistent.
        :param allow_different_commits: If False, all the metadata reports
        must come from the same commit to be considered consistent.
        :param allow_different_branches: If False, all the metadata reports
        must come from the same branch.
        :param allow_different_files: If False, all metadata objects must
        be based on the exact same file data.
        :param max_age_difference: pd.Timedelta or None. If Timedelta,
        this is the maximum difference between creation time of each
        metadata report that is allowed for the collection to be considered
        consistent.
        """
        self.allow_uncommitted = allow_uncommitted
        self.allow_different_commits = allow_different_commits
        self.allow_different_branches = allow_different_branches
        self.allow_different_files = allow_different_files
        self.max_age_difference = max_age_difference
        self.allow_different_environments = allow_different_environments

    @classmethod
    def allow_nothing(cls):
        return cls(
            allow_uncommitted=False,
            allow_different_commits=False,
            allow_different_branches=False,
            allow_different_files=False,
            allow_different_environments=False,
            max_age_difference=None)

    def validate_consistency(self, metadata):
        """
        Validates the consistency of a collection of metadata reports.
        What to consider depends on the settings specified in the constructor.

        :param metadata: Iterable of metadata reports
        :return: True if the metadata is consistent, otherwise a
        MetadataConsistencyError is raised.
        """
        self._validate_uncommitted_changes(metadata)
        self._validate_same_commit(metadata)
        self._validate_timestamp(metadata)
        self._validate_files(metadata)
        return True

    def _validate_uncommitted_changes(self, metadata):
        if not self.allow_uncommitted:
            for md in metadata:
                try:
                    if md['has_uncommitted_changes']:
                        raise MetadataConsistencyError(
                            'Uncommitted changes were present!'
                        )
                except KeyError:
                    raise MetadataConsistencyError(
                        'Metadata incomplete! No information about '
                        'uncommitted changes')

    def _validate_same_commit(self, metadata):
        if not self.allow_different_commits:
            commits = set()
            for md in metadata:
                try:
                    commits.add(md['current_commit'])
                except KeyError:
                    raise MetadataConsistencyError(
                        'Metadata incomplete! No information about '
                        'current commit.')

                if len(commits) > 1:
                    raise MetadataConsistencyError(
                        f'Commits are different: '
                        f'{commits.pop()} != {commits.pop()}')

    def _validate_same_branch(self, metadata):
        if not self.allow_different_branches:
            branches = set()
            for md in metadata:
                try:
                    branches.add(md['current_branch'])
                except KeyError:
                    raise MetadataConsistencyError(
                        'Metadata incomplete! No information about '
                        'current branch.')

                if len(branches) > 1:
                    raise MetadataConsistencyError(
                        f'Branches are different: '
                        f'{branches.pop()} != {branches.pop()}')

    def _validate_same_environment(self, metadata):
        if not self.allow_different_environments:
            envs = set()
            for md in metadata:
                try:
                    envs.add(md['conda'])
                except KeyError:
                    raise MetadataConsistencyError(
                        'Metadata incomplete! No information about '
                        'conda environment.')

                if len(envs) > 1:
                    raise MetadataConsistencyError(
                        f'Environments are different: '
                        f'{envs.pop()} != {envs.pop()}')

    def _validate_timestamp(self, metadata):
        if self.max_age_difference:
            timestamps = set()
            for md in metadata:
                try:
                    timestamps.add(pd.Timestamp(md['timestamp']))
                except KeyError:
                    raise MetadataConsistencyError(
                        'Metadata incomplete! No information about '
                        'timestamps.')

            if max(timestamps) - min(timestamps) > self.max_age_difference:
                raise MetadataConsistencyError(
                    f'The age difference between the metadata objects was '
                    f'larger than {self.max_age_difference}!'
                )

    def _validate_files(self, metadata):
        if not self.allow_different_files:
            file_data = set()
            for md in metadata:
                try:
                    # string representation of the file_data dictionary is a
                    # bit of a hack... Made a custom version just to avoid
                    # problems with differently sorted keys.
                    file_data.add(dict_to_string(md['file_data']))
                except KeyError:
                    raise MetadataConsistencyError(
                        'Metadata incomplete! No information about '
                        'file_data')

                if len(file_data) > 1:
                    raise MetadataConsistencyError(
                        'Files in data folder were different.')


def dict_to_string(d):
    s = '{'
    for key in sorted(d):
        s += key + ': ' + str(d[key]) + ', '

    return s + '}'


class MetadataConsistencyError(Exception):
    pass


def public_methods(obj):
    for f in dir(obj):
        if callable(getattr(obj, f)) and not f.startswith('_'):
            yield f, getattr(obj, f)


def save(data, path):
    md = Metadata().report()
    pd.to_pickle((data, md), path, compression='gzip')


def load(path, allow_uncommitted=False):
    data, meta_data = pd.read_pickle(path)
    v = Validator(allow_uncommitted=allow_uncommitted)
    v.validate_consistency([meta_data])
    return data
