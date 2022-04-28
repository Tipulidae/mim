import os

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from mim.config import PATH_TO_DATA
from mim.extractors.extractor import Extractor, Data, Container


PTBXL_PATH = os.path.join(PATH_TO_DATA, 'ptbxl')


def make_labels(info, sex=False, age=False, height=False, weight=False):
    assert any([sex, age, height, weight])
    labels = []
    if sex:
        labels.append("sex")
    if age:
        labels.append("age")
    if height:
        labels.append("height")
    if weight:
        labels.append("weight")

    return Data(
        info[labels].values,
        columns=labels
    )


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
