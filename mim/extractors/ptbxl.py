import os
from collections import defaultdict

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from mim.config import PATH_TO_DATA
from mim.extractors.extractor import Extractor, Data, Container


PTBXL_PATH = os.path.join(PATH_TO_DATA, 'ptbxl')


def make_labels(info, **selected):
    selected = defaultdict(bool, selected)
    allowed_labels = ['sex', 'age', 'height', 'weight']
    selected_labels = [label for label in allowed_labels if selected[label]]
    assert selected_labels

    label_dict = {
        label: Data(info[[label]].values, columns=[label])
        for label in selected_labels
    }

    return Container(label_dict)


def make_features(info, resolution='high', leads=12):
    filename = 'filename_lr' if resolution == 'low' else 'filename_hr'

    assert leads == 12 or leads == 1
    if leads == 1:
        channels = [0]
        columns = ['I']
    else:
        channels = list(range(12))
        columns = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF',
                   'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    x = np.array([
        wfdb.rdsamp(os.path.join(PTBXL_PATH, f), channels=channels)[0]
        for f in tqdm(info[filename])
    ])

    return Data(x, columns=columns)


def make_index(size=-1):
    index = (
        pd.read_csv(
            os.path.join(PTBXL_PATH, 'ptbxl_database.csv'),
            index_col='ecg_id'
        )
        .sort_values(by=['patient_id', 'recording_date'])
        .dropna(subset=['sex', 'age', 'weight', 'height'])
        .drop_duplicates(subset=['patient_id'], keep='first')
    )
    index = index[index.strat_fold < 9]
    index.patient_id = index.patient_id.astype(int)

    if 0 < size <= len(index):
        index = index.iloc[:size, :]

    return index


class PTBXL(Extractor):
    def get_data(self) -> Container:
        index = make_index(**self.index)
        n = len(index)

        data = Container(
            {
                'x': make_features(index, **self.features),
                'y': make_labels(index, **self.labels),
                'index': Data(
                    index.patient_id.values,
                    columns=['patient_id'],
                )
            },
            index=range(n),
            groups=list(range(n)),
            fits_in_memory=True,
        )

        return data
