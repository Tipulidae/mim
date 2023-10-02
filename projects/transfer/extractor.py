import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

from mim.experiments.extractor import DataWrapper, Data, Extractor
from massage import sk1718
from massage.muse_ecg import expected_lead_names
from mim.util.logs import get_logger


log = get_logger("Transfer-learning extractor")


def make_target_index(ecg_table):
    index = sk1718.make_index()
    brsm = index.loc[index.cause == 'BröstSm', :]
    brsm_ecgs = sk1718.find_index_ecgs(brsm, ecg_table)
    return brsm_ecgs


def train_val_test_split(
        index, temporal_test_percent=0.2, random_test_percent=0.1, 
        val_percent=0.2, train_percent=1.0, 
        exclude_test_aliases=False,
        ):
    dev, test = hold_out_split(
        index, temporal_test_percent, random_test_percent, exclude_test_aliases)
    train, val = development_split(dev, val_percent, train_percent)
    return train, val, test


def hold_out_split(index, temporal_test_percent, random_test_percent, 
                   exclude_test_aliases, random_state=123):
    # Sort index by admission time
    # Take the last k% of the visits
    # Remove all Aliases associated with those visits from the remainder
    # Partition the remainder on Aliases, randomly
    remainder, temporal_test = temporal_split(
        index, temporal_test_percent, exclude_test_aliases)
    splitter = GroupShuffleSplit(
        n_splits=1, 
        test_size=random_test_percent/(1-temporal_test_percent), 
        random_state=random_state
    )
    dev, random_test = next(splitter.split(remainder, groups=remainder.Alias))

    test_set = pd.concat([remainder.iloc[random_test, :], temporal_test])
    development_set = remainder.iloc[dev, :]
    return development_set, test_set
    # return development_set, remainder.iloc[random_test, :], temporal_test


def temporal_split(index, percent, exclude_test_aliases):
    n = len(index)
    test_size = int(percent * n)
    remainder_size = n - test_size
    test = index.sort_values(by=['admission_date']).iloc[remainder_size:, :]
    remainder = index.iloc[:remainder_size, :]
    if exclude_test_aliases:
        remainder = remainder.loc[~remainder.Alias.isin(test.Alias), :]
        
    return remainder, test


def development_split(index, val_percent, train_percent_of_remainder=1.0, 
                      random_state=321):
    # After splitting into train/val sets, we return only a subset of the index 
    # depending on the train_percent_of_remainder parameter.
    random_aliases = (
        pd.Series(index.Alias.unique())
        .sort_values()
        .sample(frac=1, random_state=random_state)
    )
    n = len(random_aliases)
    val_size = int(n * val_percent)
    train_size = int((n - val_size) * train_percent_of_remainder)
    log.debug(f"{n=}, {val_size=}, {train_size=}")
    val_aliases = random_aliases.iloc[:val_size]
    train_aliases = random_aliases.iloc[val_size:val_size + train_size]
    val = index.loc[index.Alias.isin(val_aliases), :]
    train = index.loc[index.Alias.isin(train_aliases), :]

    return train, val


def make_source_labels(index):
    # I don't really trust the age and sex provided by the ECG itself, so 
    # instead I will look at "liggaren" for this information. The age in 
    # particular is sometimes wrong for the ECG. I will use the age at the 
    # first ED visit to approximate the birthday, and then use 
    # birthday + ecg timestamp to calculate the (approximate) age.
    # I could get a more accurate estimate of the birthday by looking at the 
    # age at more ED-visits, but this should be good enough, and the error is 
    # bounded by one year anyway, assuming the information in liggaren is 
    # correct. 
    liggaren = sk1718.make_index()
    liggaren = (
        liggaren
        .sort_values(by=['Alias', 'admission_date'])
        .drop_duplicates(subset=['Alias'], keep='first')
        .set_index('Alias')
        .loc[:, ['admission_date', 'age', 'sex']]
    )
    # Adding 0.5 years should give an unbiased estimate of the true age, 
    # since age in years is rounded down.
    liggaren['birthday'] = (
        liggaren.admission_date - 
        ((liggaren.age + 0.5) * 365.125).astype('timedelta64[D]')
    )
    index = index.join(liggaren[['birthday', 'sex']], on='Alias', how='left')
    index['age'] = (
        (index.ecg_date - index.birthday).dt.days / 365.125).astype(int)
    index['sex'] = index.sex.map({'M': 0, 'F': 1})
    return index


def make_source_index(
        exclude_test_aliases=True, exclude_test_ecgs=True,
        exclude_val_aliases=True, exclude_val_ecgs=True,
        exclude_train_aliases=False, exclude_train_ecgs=False):
    source = sk1718.make_ecg_table()  # We start with all the ECGs
    target = make_target_index(source)  # All target ECGs before any splits
    train, val, test = train_val_test_split(target)
    source = source.reset_index()

    def exclude(column, df, msg):
        n1 = len(source)
        new_source = source.loc[~source[column].isin(df[column]), :]
        n2 = len(new_source)
        log.info(f"Excluded {n1-n2} ECGs from the {msg}")
        return new_source

    # Exclude some of the ecgs from the source dataset
    if exclude_test_aliases:
        source = exclude('Alias', test, 'test set')
    elif exclude_test_ecgs:
        source = exclude('ecg_id', test, 'test set')
    if exclude_val_aliases:
        source = exclude('Alias', val, 'validation set')
    elif exclude_val_ecgs:
        source = exclude('ecg_id', val, 'validation set')
    if exclude_train_aliases:
        source = exclude('Alias', train, 'training set')
    elif exclude_train_ecgs:
        source = exclude('ecg_id', train, 'training set')
    
    return (
        source
        .sort_values(by=['ecg_id'])
        .loc[:, ['ecg_id', 'Alias', 'ecg_date']]
    )


def make_ecg_data(ecgs, mode):
    with h5py.File('/projects/air-crypt/axel/sk1718_ecg.hdf5', 'r') as f:
        data = np.array([f[mode][ecg] for ecg in tqdm(ecgs)])
    
    return data


class TargetTask(Extractor):
    def get_development_data(self) -> DataWrapper:
        ecg_table = sk1718.make_ecg_table()
        brsm_ecgs = make_target_index(ecg_table)
        train, val, _ = train_val_test_split(brsm_ecgs, **self.index)
        dev = pd.concat([train, val])

        # TODO: not the real target, fix this later!
        y = dev.sex.map({'M': 0, 'F': 1})
        

        return DataWrapper(
            features=Data(
                data=make_ecg_data(dev.ecg_id, self.features['ecg_mode']),
                columns=expected_lead_names,
            ),
            labels=Data(y.values, columns=['sex']),
            index=Data(dev.ecg_id.values, columns=['ecg_id']),
            groups=dev.Alias.values,
            predefined_splits=len(train)*[-1] + len(val)*[0],
            fits_in_memory=True
        )

    def get_test_data(self) -> DataWrapper:
        ecg_table = sk1718.make_ecg_table()
        brsm_ecgs = make_target_index(ecg_table)
        _, _, test = train_val_test_split(brsm_ecgs, **self.index)

        # TODO: not the real target, fix this later!
        y = test.sex.map({'M': 0, 'F': 1})

        return DataWrapper(
            features=Data(
                data=make_ecg_data(test.ecg_id, self.features['ecg_mode']),
                columns=expected_lead_names,
            ),
            labels=Data(y.values, columns=['sex']),
            index=Data(test.ecg_id.values, columns=['ecg_id']),
            groups=test.Alias.values,
            fits_in_memory=True
        )


class SourceTask(Extractor):
    def get_development_data(self) -> DataWrapper:
        index = make_source_index(**self.index)

        # TODO: parameterize this!
        y = make_source_labels(index)
        
        return DataWrapper(
            features=Data(
                data=make_ecg_data(index.ecg_id, self.features['ecg_mode']),
                columns=expected_lead_names,
            ),
            labels=Data(y.sex.data, columns=['sex']),
            index=Data(index.ecg_id.values, columns=['ecg_id']),
            groups=index.Alias.values,
            fits_in_memory=True
        )

    def get_test_data(self) -> DataWrapper:
        raise NotImplementedError
