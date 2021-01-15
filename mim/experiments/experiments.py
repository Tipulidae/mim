import os
from pathlib import Path
from typing import Any, NamedTuple

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from mim.cross_validation import CrossValidationWrapper, ChronologicalSplit
from mim.config import PATH_TO_TEST_RESULTS, HyperParams
from mim.model_wrapper import RandomForestClassifier


class Experiment(NamedTuple):
    description: str
    nickname: str = None
    extractor: Any = None
    index: Any = None
    features: Any = None
    labels: Any = None
    post_processing: Any = {'imputation'}
    params: Any = HyperParams.P0
    algorithm: Any = RandomForestClassifier
    wrapper: Any = None
    predict_only: bool = False
    cv: Any = KFold
    cv_args: dict = {}
    scoring: Any = roc_auc_score
    hold_out: Any = ChronologicalSplit
    hold_out_size: float = 0

    def get_data(self):
        """
        Uses the extractor and specifications to create the X and y data
        set.

        :return: Data object
        """
        specification = {
            'index': self.index,
            'features': self.features,
            'labels': self.labels,
            'processing': self.post_processing
        }
        data = self.extractor(specification=specification).get_data()
        splitter = self.hold_out(test_size=self.hold_out_size)
        develop_index, test_index = next(splitter.split(data))
        return data.split(develop_index, test_index)
        # return self.extractor(specification=specification).get_data()

    @property
    def cross_validation(self):
        return CrossValidationWrapper(self.cv, **self.cv_args)

    @property
    def is_binary(self):
        return self.labels['is_binary']

    @property
    def classifier(self):
        default_params = {}
        ps = self.params
        if isinstance(ps, HyperParams):
            ps = self.params.value

        ps = {**default_params, **ps}
        if self.wrapper is not None:
            ps['wrapper'] = self.wrapper

        if 'random_state' not in ps:
            ps['random_state'] = 123

        return self.algorithm(**ps)

    @property
    def name(self):
        if hasattr(self, '_name_'):
            return self._name_
        else:
            return self.nickname

    @property
    def result_path(self):
        return os.path.join(
            PATH_TO_TEST_RESULTS,
            f'{self.__class__.__name__}_{self.name}.experiment')

    @property
    def is_done(self):
        file = Path(self.result_path)
        return file.is_file()


def result_path(xp):
    return os.path.join(
        PATH_TO_TEST_RESULTS,
        f'{xp.__class__.__name__}_{xp.name}.experiment')
