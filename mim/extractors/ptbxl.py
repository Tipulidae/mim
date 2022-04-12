import ast

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from mim.config import PATH_TO_DATA
from mim.extractors.extractor import Extractor, Data, Container


class PTBXL(Extractor):
    def get_data(self) -> Container:
        path = PATH_TO_DATA+'/ptbxl/'
        info = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        if self.index['size'] == 'XS':
            info = info.iloc[:200, :]

        # holding out the last fold
        info = info[info.strat_fold < 9]

        info = (
            info
            .sort_values(by=['patient_id', 'recording_date'])
            .drop_duplicates(subset=['patient_id'], keep='first')
        )
        info.scp_codes = info.scp_codes.apply(ast.literal_eval)
        info['bmi'] = info.weight / ((info.height/100)**2)
        info = info.dropna(subset=['bmi'])
        y = (info['bmi'] >= 30).values
        x = np.array([wfdb.rdsamp(path+f)[0] for f in tqdm(info.filename_lr)])

        # n = len(y)
        num_patients, sample_frequency, num_leads = x.shape
        groups = np.arange(num_patients)

        index_data = info.patient_id.values

        if 'leads' in self.index and self.index['leads'] == 'single':
            y = repeat_k_times(y, k=num_leads)
            groups = repeat_k_times(np.arange(num_patients), k=num_leads)
            index_data = repeat_k_times(index_data, k=num_leads)

            x = x.transpose((0, 2, 1)).reshape((-1, sample_frequency))
            x = np.expand_dims(x, axis=-1)  # Otherwise tf complains

        data = Container(
            {
                'x': Data(x),
                'y': Data(y),
                'index': Data(
                    index_data,
                    columns=['patient_id'],
                )
            },
            index=range(len(y)),
            groups=groups,
            fits_in_memory=True,
        )

        return data


def repeat_k_times(x, k):
    """
    Example: repeat_k_times([1, 2, 3, 4], 3]) ->
    [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    """
    return np.tile(x, (k, 1)).ravel(order='F')
