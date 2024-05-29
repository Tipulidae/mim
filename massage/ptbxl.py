import os
import ast
from copy import deepcopy

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

from mim.config import PATH_TO_DATA
from mim.cache.decorator import cache


PTBXL_PATH = os.path.join(PATH_TO_DATA, 'ptbxl')


def compute_label_aggregations(info):
    df = deepcopy(info)
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    df['scp_codes_len'] = df.scp_codes.apply(len)
    df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))
    return df


def select_data(info, min_samples=1):
    YY = compute_label_aggregations(info)

    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()
    counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
    counts = counts[counts > min_samples]
    YY.all_scp = YY.all_scp.apply(
        lambda x: list(set(x).intersection(set(counts.index.values))))
    YY['all_scp_len'] = YY.all_scp.apply(len)
    Y = YY[YY.all_scp_len > 0]
    mlb.fit(Y.all_scp.values)
    y = mlb.transform(Y.all_scp.values)

    return pd.DataFrame(y, index=Y.index, columns=mlb.classes_)


def load_info():
    index = (
        pd.read_csv(
            os.path.join(PTBXL_PATH, 'ptbxl_database.csv'),
            index_col='ecg_id'
        )
        .sort_values(by=['patient_id', 'recording_date'])
    )
    index.patient_id = index.patient_id.astype(int)
    return index


def make_labels(index):
    info = pd.read_csv(
        os.path.join(PTBXL_PATH, 'ptbxl_database.csv'),
        index_col='ecg_id'
    )
    labels = select_data(info)
    return labels.loc[index.index, :]


@cache
def process_ptbxl_ecgs(info, resolution='high', leads=12):
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
