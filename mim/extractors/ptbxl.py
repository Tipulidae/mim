import ast

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from mim.config import PATH_TO_DATA
from mim.extractors.extractor import Extractor, DataProvider


class PTBXL(Extractor):
    def get_data_provider(self, dp_kwargs) -> DataProvider:
        path = PATH_TO_DATA+'/ptbxl/'
        info = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        if self.index['size'] == 'XS':
            info = info.iloc[:200, :]

        info.scp_codes = info.scp_codes.apply(ast.literal_eval)

        statements = pd.read_csv(path+'scp_statements.csv', index_col=0)

        def is_mi(scp_dict):
            for scp in scp_dict:
                if statements.loc[scp].diagnostic_class == 'MI':
                    return 1
            return 0

        y = info.scp_codes.apply(is_mi)
        x = np.array([wfdb.rdsamp(path+f)[0] for f in tqdm(info.filename_lr)])

        x = x[info.strat_fold < 9]
        y = y[info.strat_fold < 9]

        return None  # XXX: This needs to be fixed
