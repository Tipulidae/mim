import pandas as pd
import numpy as np
from scipy import sparse

import setuptools
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from mim.cross_validation import CrossValidationWrapper
from mim.experiments.extractor import Extractor, DataWrapper, Data, Container
from mim.util.logs import get_logger
from massage import rsvd_plus
from mim.cache.decorator import cache

log = get_logger("RSVD Extractor")

@cache
def make_all_data_index_test(days):
    all_data = rsvd_plus.make_data(days)
    all_data['vaccination_date'] = pd.to_datetime(all_data['vaccination_date'])
    all_data['Troponin'] = all_data['P-Troponin I NY'] + all_data['P-Troponin T']
    all_data.drop(['P-Troponin I NY', 'P-Troponin T'], axis=1, inplace=True)
    all_data = all_data.set_index(["Alias", "vaccination_date"])
    end_cols = ["Age", "Male", "Female", "Dose_1", "Dose_2", "Dose_more_3", "Pfizer", "Moderna", "AstraZeneca"]
    data_end = all_data[end_cols]
    data_start = all_data[all_data.columns.difference(end_cols)]
    data_start = 1 - 1 / 2 ** data_start
    data_start.replace([np.inf, -np.inf], 1, inplace=True)
    normalized_age = (data_end["Age"] - data_end["Age"].min()) / (data_end["Age"].max() - data_end["Age"].min())
    data_end['Age'] = normalized_age
    data_new = pd.merge(data_start, data_end, left_index=True, right_index=True)
    sparse_matrix = sparse.csr_matrix(data_new.values)
    sparse_df = pd.DataFrame.sparse.from_spmatrix(
        sparse_matrix,
        index=data_new.index,
        columns=data_new.columns
    )
    return sparse_df


@cache
def data_management_pre_wrap(simple_fake=None, drop_name=None):
    normal_data = make_all_data_index_test(-20).sparse.to_dense()
    signal_data = make_all_data_index_test(20).sparse.to_dense()
    if simple_fake:
        fake = simple_fake[0]  # [0, 0.5, 0.75, 0.875]
        probability_normal = simple_fake[1]
        probability_signal = simple_fake[2]
        normal_data['fake'] = pd.NA
        normal_data['fake'] = normal_data['fake'].apply(lambda x: np.random.choice(fake, p=probability_normal))
        signal_data['fake'] = pd.NA
        signal_data['fake'] = signal_data['fake'].apply(lambda x: np.random.choice(fake, p=probability_signal))
    normal_data, signal_data = remove_access_features(normal_data, signal_data)
    normal_data["Postvaccination_index"] = 0
    signal_data["Postvaccination_index"] = 1

    all_data = pd.concat([normal_data, signal_data])

    all_data = all_data.set_index(["Postvaccination_index"], append=True, drop=False)
    all_data = all_data.rename(columns={"Postvaccination_index": "Postvaccination"})
    non_zero_cols = \
        ["A-B", "C-D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S-T", "U", "V-Y",
         "Z", "Age", "Male", "Female", "Dose_1", "Dose_2", "Dose_more_3", "Pfizer", "Moderna", "AstraZeneca"]
    zero_data = all_data.loc[~(all_data.loc[:, non_zero_cols] != 0).any(axis=1)].reset_index()
    drop_index = zero_data[zero_data.groupby(["Alias", "vaccination_date"])[['Alias', 'vaccination_date']]
                           .transform('size') > 1].set_index(["Alias", "vaccination_date", "Postvaccination"]).index
    all_data = all_data.drop(index=drop_index)

        # ny_df = all_data[all_data["NY"] < 0.25].reset_index()
        # drop_ny_index = ny_df[ny_df.groupby(["Alias", "vaccination_date"])[['Alias', 'vaccination_date']].
        #                       transform('size') > 1].set_index(["Alias", "vaccination_date", "Postvaccination"]).index
        # all_data = all_data.drop(index=drop_ny_index)

    if drop_name:
        all_data = all_data.drop(columns=drop_name)

    sparse_matrix = sparse.csr_matrix(all_data.values)
    sparse_df = pd.DataFrame.sparse.from_spmatrix(
        sparse_matrix,
        index=all_data.index,
        columns=all_data.columns
    )
    return sparse_df


def remove_access_features(normal_df, signal_df):
    difference1 = list((set(list(normal_df)) - set(list(signal_df))))
    difference2 = list((set(list(signal_df)) - set(list(normal_df))))
    normal_df = normal_df.drop(columns=difference1)
    signal_df = signal_df.drop(columns=difference2)
    return normal_df, signal_df


class PredictionTasks(Extractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        all_data = data_management_pre_wrap(simple_fake=self.simple_fake, drop_name=self.drop_name)\
            .sparse.to_dense().sample(frac=1, random_state=44)
        data = DataWrapper(
            features=Data(all_data.iloc[:, :-1].values,
                          columns=list(all_data.iloc[:, :-1])),
            labels=Data(all_data.iloc[:, -1:].values,
                        columns=['Postvaccination']),
            index=Data(all_data.index,
                       columns=['Alias', 'vaccination_date', 'Postvaccination_index']),
            groups=all_data.reset_index().Alias.values
        )
        hold_out_splitter = CrossValidationWrapper(
            GroupShuffleSplit(test_size=0.25, random_state=44)
        )
        development_data, test_data = next(hold_out_splitter.split(data))
        self.development_data = development_data
        self.test_data = test_data

    def get_development_data(self) -> DataWrapper:
        log.debug("Making development DataWrapper")
        return self.development_data

    def get_test_data(self) -> DataWrapper:
        log.debug("Making test DataWrapper")
        return self.test_data



class SelfControl(Extractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        normal_data = make_all_data(-30)
        signal_data = make_all_data(30)
        normal_data_dev, _ = remove_access_features(normal_data, signal_data)
        # labels = [1] * normal_data.shape[0]
        # normal_data_dev, _, _, _ = train_test_split(normal_data, labels, test_size=0.33)
        self.normal_data_dev = normal_data_dev

    def get_development_data(self) -> DataWrapper:
        log.debug("Making DataWrapper")
        return DataWrapper(
            features=Data(self.normal_data_dev.values, columns=list(self.normal_data_dev)),
            labels=Container({'Med': Data(self.normal_data_dev.iloc[:, :-3].values,
                                          columns=list(self.normal_data_dev.iloc[:, :-3])),
                              'Age': Data(self.normal_data_dev['Age'].values, columns=['Age']),
                              'Gender': Data(self.normal_data_dev.iloc[:, -2:].values, columns=['Male', 'Female'])}),
            index=Data(range(len(self.normal_data_dev)), columns=['foo'])
        )

    def get_test_data(self) -> DataWrapper:
        log.debug("Making test DataWrapper")

        return DataWrapper(
            features=Data(self.normal_data_test.values, columns=list(self.normal_data_test)),
            labels=Container({'Med': Data(self.normal_data_test[:, :-3], columns=list(self.normal_data_test[:, :-3])),
                              'Age': Data(self.normal_data_test['Age'].values, columns=['Age']),
                              'Gender': Data(self.normal_data_test.iloc[:, -2:].values, columns=['Male', 'Female'])}),
            index=Data(range(len(self.normal_data_test)), columns=['foo'])
        )

@cache
def make_all_data(days):
    all_data = rsvd_plus.make_data(days)
    all_data["FödelseårOchMånad"] = pd.to_datetime(all_data["FödelseårOchMånad"], format="%Y%m")
    age = pd.to_datetime("202201", format="%Y%m") - all_data["FödelseårOchMånad"]
    all_data["Age"] = (age.dt.days / 365.25).astype(int)
    all_data["Male"] = pd.get_dummies(all_data["Kön"])["M"]
    all_data["Female"] = pd.get_dummies(all_data["Kön"])["K"]
    all_data = all_data.drop(
        ["FödelseårOchMånad", "Alias", "vaccination_date", "vaccine_product", "Kön", "dose_number"], axis=1)
    data_end = all_data.iloc[:, -3:]
    data_start = all_data.iloc[:, :-3]
    data_start = 1 - 1 / 2 ** data_start
    data_start.replace([np.inf, -np.inf], 1, inplace=True)
    data_end["Age"] = (data_end["Age"] - data_end["Age"].min()) / (data_end["Age"].max() - data_end["Age"].min())
    data_new = pd.merge(data_start, data_end, left_index=True, right_index=True)
    return data_new


