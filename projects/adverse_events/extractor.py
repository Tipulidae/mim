
import pandas as pd
import numpy as np

from mim.experiments.extractor import Extractor, DataWrapper, Data, Container
from mim.util.logs import get_logger


log = get_logger("RSVD Extractor")


class SelfControl(Extractor):
    def get_development_data(self) -> DataWrapper:
        log.debug("Loading csv file")

        all_data = pd.read_csv(
            "/mnt/air-crypt/air-crypt-esc-trop/thomas/MASSAGED_DATA/AutoencoderProject/savedDataSets/"
            "20230224-test500.csv",
            compression='gzip')
        all_data["FödelseårOchMånad"] = pd.to_datetime(all_data["FödelseårOchMånad"], format="%Y%m")
        age = pd.to_datetime("202201", format="%Y%m") - all_data["FödelseårOchMånad"]
        all_data["Age"] = (age.dt.days / 365.25).astype(int)
        all_data["Male"] = pd.get_dummies(all_data["Kön"])["M"]
        all_data["Female"] = pd.get_dummies(all_data["Kön"])["K"]
        all_data = all_data.drop(
            ["FödelseårOchMånad", "vaccination_date", "Alias", "vaccine_product", "Kön", "dose_number"], axis=1)
        data_end = all_data.iloc[:, -3:]
        data_start = all_data.iloc[:, :-3]
        data_start = 1 - 1 / 2 ** data_start
        data_start.replace([np.inf, -np.inf], 1, inplace=True)
        data_end["Age"] = (data_end["Age"] - data_end["Age"].min()) / (data_end["Age"].max() - data_end["Age"].min())
        data_new = pd.merge(data_start, data_end, left_index=True, right_index=True)
        log.debug("Making DataWrapper")
        return DataWrapper(
            features=Data(data_new.values, columns=list(data_new)),
            labels=Container({'Med': Data(data_start.values, columns=list(data_start)),
                              'Age': Data(data_end['Age'].values, columns=['Age']),
                              'Gender': Data(all_data.iloc[:, -2:].values, columns=['Male', 'Female'])}),
            index=Data(range(len(all_data)), columns=['foo'])
        )

    def get_test_data(self) -> DataWrapper:
        raise NotImplementedError
