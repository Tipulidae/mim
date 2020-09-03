import os
from enum import Enum

ROOT_PATH = os.path.join(os.path.dirname(__file__), '..')
PATH_TO_DATA = os.path.join(ROOT_PATH, 'data')
PATH_TO_TEST_RESULTS = os.path.join(ROOT_PATH, 'data', 'test_results')


class HyperParams(Enum):
    P0 = {
        'n_estimators': 1000,
        'bootstrap': False,
    }

    P1 = {
        'n_estimators': 1000,
        'bootstrap': True,
        'min_samples_leaf': 20,
        'max_features': 5
    }

    Small = {
        'n_estimators': 100,
        'random_state': 10
    }
