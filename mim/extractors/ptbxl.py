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

        info.scp_codes = info.scp_codes.apply(ast.literal_eval)
        info['bmi'] = info.weight / ((info.height/100)**2)
        info = info.dropna(subset=['bmi'])
        y = info['bmi'] >= 30
        x = np.array([wfdb.rdsamp(path+f)[0] for f in tqdm(info.filename_lr)])

        # Holding out the last fold
        x = x[info.strat_fold < 9]
        y = y[info.strat_fold < 9]

        data = Container(
            {
                'x': Data(x),
                'y': Data(y.values),
            },
            index=range(len(y)),
        )

        return data
