import os
from pathlib import Path
from typing import Any, NamedTuple

from sklearn.metrics import roc_auc_score

from mim.cross_validation import KFold
from mim.config import PATH_TO_TEST_RESULTS, HyperParams
from mim.model_wrapper import RandomForestClassifier


class Experiment(NamedTuple):
    description: str
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

    def get_data(self):
        """
        Uses the extractor and specifications to create the X and y data
        set.

        :return: Tuple X, X_validate, y, where X, y comes from the
        extractor. The X_validate is None, but is needed in the return value
        to conform with a different type of experiment in which a different
        data set is generated for validation.
        """
        specification = {
            'index': self.index,
            'features': self.features,
            'labels': self.labels,
            'processing': self.post_processing
        }
        X, y = self.extractor(specification=specification).get_data()

        X_validate = None
        return X, X_validate, y

    @property
    def cross_validation(self):
        return self.cv(**self.cv_args)

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

        return self.algorithm(**ps)

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
