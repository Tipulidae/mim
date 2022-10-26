import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from mim.experiments.extractor import Data, RaggedData, Extractor, \
    DataWrapper, Container
from mim.util.logs import get_logger
from mim.cache.decorator import cache
from massage import sk1718


log = get_logger("Skåne-1718 extractor")


def make_basic_features(brsm):
    data = brsm.loc[:, ['age', 'sex']].copy()
    # data['age'] = (data['age'] - 50) / 100
    data['male'] = (data.sex == 'M').astype(int)
    data['female'] = (data.sex == 'F').astype(int)

    return Data(
        data=data[['age', 'male', 'female']].values,
        columns=['age', 'male', 'female']
    )


def helsingborg_partition(brsm):
    """
    Given a dataframe with columns Alias and hospital, I want to return a
    mapping (pd.Series) from Alias to the numbers 0, 1 and 2:
    0 - Patient has never been to Helsingborg
    1 - Current visit is at Helsingborg
    2 - Current visit is not Helsingborg, but patient has other visits
    to Helsingborg
    """
    # All events at Helsingborg...
    hbg = brsm.set_index('Alias').hospital == 'Helsingborgs lasarett'

    # All events for patients who at some point were at Helsingborg
    hbg_and_related = hbg.groupby('Alias').any().reindex(brsm.Alias)

    # All events not at Helsingborg, but where the patient has been
    # there at some point (These will be excluded)
    related = hbg ^ hbg_and_related

    # Each event is now mapped to 0 for development, 1 for test, and
    # 2 for events that should not be included (non-hbg events for
    # patients with at least 1 other hbg event)
    split = hbg.astype(int) + related.map({False: 0, True: 2})
    return split


@cache
def make_brsm():
    index = sk1718.make_index()
    index = index.loc[index.cause == 'BröstSm', :]
    mace = sk1718.make_mace(index)
    data = index.set_index(['Alias', 'KontaktId']).join(mace)
    data['admission_index'] = range(len(data))
    data.admission_date = data.admission_date.dt.floor('D')

    return data.reset_index()


def make_lisa_features(
        brsm,
        onehot=True,
        bin_income=False,
        family=False,
        education=False,
        occupation=False,
        income=False,
):
    lisa = sk1718.lisa(brsm, bin_income=bin_income)
    cols = ['lisa_missing']
    if not any([family, education, occupation, income]):
        cols = list(lisa)

    if family:
        cols.extend([
            'children_aged_0_3',
            'children_aged_4_6',
            'children_aged_7_10',
            'children_aged_11_15',
            'children_aged_16_17',
            'citizenship_eu15',
            'citizenship_eu28',
            'marital_status',
        ])
    if education:
        cols.extend([
            'education_duration',
            'education_field',
            'education_level',
            'education_level_old',
            'education_type',
            'graduation_decade',
        ])
    if occupation:
        cols.extend([
            'occupation_code',
            'occupation_type',
            'occupational_status',
            'socioeconomic_class',
            'socioeconomic_group',
        ])
    if income:
        cols.extend([
            'capital_income',
            'disposable_income',
            'disposable_income_family_v1',
            'disposable_income_family_v2',
            'early_retirement_benefit',
            'housing_benefit',
            'parental_benefit',
            'political_benefit',
            'received_early_retirement_benefit',
            'received_sickness_benefit',
            'retirement_pension',
            'sickness_and_rehab_benefit',
            'sickness_benefit',
            'sickness_benefit_days',
            'sickness_pension_days',
            'social_benefit',
            'unemployment_benefit',
            'unemployment_days',
        ])
    # log.debug(f"{cols=}")
    lisa = lisa[cols]

    if onehot:
        categorical_cols = lisa.select_dtypes(include=['category']).columns
        remaining_cols = lisa.columns.difference(categorical_cols)
        ohe = OneHotEncoder(sparse=False)
        lisa = pd.concat([
            pd.DataFrame(
                ohe.fit_transform(lisa[categorical_cols]),
                index=lisa.index,
                columns=ohe.get_feature_names_out(),
            ),
            lisa[remaining_cols]
        ], axis=1).astype(float)

    return lisa


class Base(Extractor):
    def get_data(self, brms) -> DataWrapper:
        raise NotImplementedError

    def get_labels(self, brsm):
        if 'outcome' in self.labels:
            outcome = self.labels['outcome']
        else:
            outcome = 'ACS'

        if outcome == 'age':
            return Data(
                brsm['age'].values,
                columns=['Age']
            )
        elif outcome == 'male':
            return Data(
                (brsm.sex == 'M').astype(int).values,
                columns=['Sex']
            )
        elif outcome == 'ACS':
            outcome_columns = ['I200', 'I21', 'I22', 'death']
            y = brsm.loc[:, outcome_columns].any(axis=1)
            return Data(y.values, columns=['ACS'])
        elif outcome == 'BAD':  # For want of better label
            # ACS + pulmonary embolism + aorta dissection + death
            outcome_columns = ['I200', 'I21', 'I22', 'I26', 'I71', 'death']
            y = brsm.loc[:, outcome_columns].any(axis=1)
            return Data(y.values, columns=['BAD'])

    def _dev_test_split(self):
        # The idea is to use any events at Helsingborg as the test (held-out)
        # set. But some patients have been to multiple hospitals, including
        # Helsingborg - those visits should then be excluded.

        # Load the index data
        # brsm = sk1718.load_outcomes_from_ab_brsm()
        brsm = make_brsm()

        # Create the whole dataset
        data = self.get_data(brsm)

        # Now we partition the data into our development/test/exclude sets
        split = helsingborg_partition(brsm)
        dev, test, *_ = data.lazy_partition(split)
        return dev, test

    def get_development_data(self) -> DataWrapper:
        dev, _ = self._dev_test_split()
        return dev

    def get_test_data(self) -> DataWrapper:
        _, test = self._dev_test_split()
        return test


class Flat(Base):
    def get_data(self, brsm) -> DataWrapper:
        x_dict = {}
        if 'history' in self.features:
            patient_history = sk1718.summarize_patient_history(
                brsm, **self.features['history']
            )
            x_dict['history'] = Data(
                patient_history.values,
                columns=list(patient_history.columns),
            )

        if 'basic' in self.features:
            x_dict['basic'] = make_basic_features(brsm)

        if 'lisa' in self.features:
            lisa = make_lisa_features(brsm, **self.features['lisa'])
            x_dict['lisa'] = Data(lisa.values, columns=list(lisa.columns))

        data = DataWrapper(
            features=Container(x_dict),
            labels=self.get_labels(brsm),
            index=Data(brsm.KontaktId.values, columns=['KontaktId']),
            groups=brsm.Alias.values,
            fits_in_memory=True
        )
        return data


class Ragged(Base):
    def get_data(self, brsm) -> DataWrapper:
        data = sk1718.staggered_patient_history(brsm, **self.features)
        data_cols = ['years_ago'] + list(data.filter(regex='(ICD)|(ATC)|(OP)'))

        data['data_index'] = range(len(data))
        row_starts = (
            data.groupby('admission_index')[['data_index']]
            .first().join(brsm, how='outer')
            .fillna(method='bfill').data_index.astype(int)
        )
        row_ends = (
            data.groupby('admission_index')[['data_index']]
            .last().join(brsm, how='outer')
            .fillna(method='ffill').data_index.astype(int) + 1
        )

        data = DataWrapper(
            features=RaggedData(
                data[data_cols].astype(float).values,
                index=range(len(brsm)),
                slices=list(zip(row_starts, row_ends)),
                columns=data_cols
            ),
            labels=self.get_labels(brsm),
            index=Data(brsm.KontaktId.values, columns=['KontaktId']),
            groups=brsm.Alias.values,
            fits_in_memory=True
        )
        return data
