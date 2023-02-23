from collections import defaultdict

import pandas as pd
import numpy as np

from mim.experiments.extractor import Extractor, DataWrapper, Data
from massage.rsvd_plus import make_index, make_data
from mim.util.logs import get_logger


log = get_logger("RSVD Extractor")

class SelfControl(Extractor):
    def get_development_data(self) -> DataWrapper:
        log.debug("Loading csv file")
        vector = pd.read_csv(
            "/mnt/air-crypt/air-crypt-esc-trop/thomas/MASSAGED_DATA/AutoencoderProject/savedDataSets/selfNormal230210.csv",
            compression='gzip')
        all_Data = vector
        dataEND = all_Data.iloc[:, -3:]
        dataSTART = all_Data.iloc[:, :-3]
        dataSTART = 1 - 1/2**dataSTART
        dataSTART.replace([np.inf, -np.inf], 1, inplace=True)
        dataEND["Age"] = (dataEND["Age"] - dataEND["Age"].min()) / (dataEND["Age"].max() - dataEND["Age"].min())
        vector = pd.merge(dataSTART, dataEND, left_index=True, right_index=True)

        log.debug("Making DataWrapper")
        return DataWrapper(
            features=Data(vector.values, columns=list(vector)),
            labels=Data(vector.values, columns=list(vector)),
            index=Data(range(len(vector)), columns=['foo'])
        )

    def get_test_data(self) -> DataWrapper:
        raise NotImplementedError
