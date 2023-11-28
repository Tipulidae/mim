import os
from collections import defaultdict

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from mim.config import PATH_TO_DATA
from mim.experiments.extractor import Extractor, DataWrapper, Data


PTBXL_PATH = os.path.join(PATH_TO_DATA, 'ptbxl')


def make_labels(info, **selected):
    selected = defaultdict(bool, selected)
    allowed_labels = ['sex', 'age', 'height', 'weight']
    selected_labels = [label for label in allowed_labels if selected[label]]
    if not selected_labels:
        raise ValueError(f'Must select at least one of {allowed_labels}')

    return info[selected_labels]


def make_features(info, resolution='high', leads=12):
    filename = 'filename_lr' if resolution == 'low' else 'filename_hr'

    if leads == 1:
        channels = [0]
        columns = ['I']
    elif leads == 8:
        channels = [6, 7, 8, 9, 10, 11, 0, 1]
        columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'I', 'II']
    elif leads == 12:
        channels = list(range(12))
        columns = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF',
                   'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    else:
        raise ValueError(f'leads must be either 1, 8 or 12, was {leads}')

    x = np.array([
        wfdb.rdsamp(os.path.join(PTBXL_PATH, f), channels=channels)[0]
        for f in tqdm(info[filename])
    ])

    return x, columns


def _load_ptbxl_index_info():
    index = (
        pd.read_csv(
            os.path.join(PTBXL_PATH, 'ptbxl_database.csv'),
            index_col='ecg_id'
        )
        .sort_values(by=['patient_id', 'recording_date'])
        .dropna(subset=['sex', 'age', 'weight', 'height'])
        .drop_duplicates(subset=['patient_id'], keep='first')
    )
    index.patient_id = index.patient_id.astype(int)
    return index


def make_development_index(size=-1):
    index = _load_ptbxl_index_info()
    index = index[index.strat_fold < 9]
    if 0 < size <= len(index):
        index = index.iloc[:size, :]

    return index


def make_test_index():
    index = _load_ptbxl_index_info()
    index = index[index.strat_fold == 9]
    return index


class PTBXL(Extractor):
    def get_development_data(self) -> DataWrapper:
        index = make_development_index(**self.index)
        return self._make_data(index)

    def get_test_data(self) -> DataWrapper:
        index = make_test_index()
        return self._make_data(index)

    def _make_data(self, index):
        ecgs, columns = make_features(index, **self.features)
        labels = make_labels(index, **self.labels)
        return DataWrapper(
            features=Data(data=ecgs, columns=columns),
            # Hack to make it work before I go home
            labels=Data(data=labels.values, columns=list(labels)),
            index=Data(index.patient_id.values,
                       columns=['patient_id']),
            fits_in_memory=True
        )
