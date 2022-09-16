import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit

from mim.experiments.extractor import Data, RaggedData, Extractor, \
    DataWrapper, Container
from mim.cross_validation import CrossValidationWrapper
from mim.util.logs import get_logger
from mim.util.metadata import load
from massage import sk1718


log = get_logger("Skåne-1718 extractor")


def load_and_clean_ab_brsm():
    data = pd.read_csv(
        '/mnt/air-crypt/air-crypt-esc-trop/andersb/scratch/brsm-U.csv',
        parse_dates=['Vardkontakt_InskrivningDatum'],
        dtype={'blood-P-Krea': str, 'blood-P-Glukos': str}
    )

    # Fix the lab-values
    lab_value_cols = [
        'blood-B-Hb',
        'blood-P-TnT',
        'blood-P-Krea',
        'blood-P-Glukos'
    ]
    lab_values = data.loc[:, lab_value_cols].applymap(
        sk1718.lab_value_to_float)
    lab_values = lab_values.dropna(how='any')

    # Drop all rows where we don't have lab-values
    data = data.loc[lab_values.index, :]
    return data, lab_values


def make_medicine_features(data, use_180d=True, use_1825d=True):
    def remove_prefix(x):
        return x[4:]

    pattern = []
    if use_180d:
        pattern.append('(med-180d)')
    if use_1825d:
        pattern.append('(med-1825d)')

    med = data.filter(regex='|'.join(pattern))
    return Data(
        med.values,
        columns=[remove_prefix(col) for col in med.columns]
    )


def make_acs_labels(data, time_span='30d', use_sv=True, use_ov=False,
                    use_death=True):
    acs_icd_codes = ['I200', 'I21', 'I22']
    columns = []
    if use_sv:
        columns += [x+'-SV' for x in acs_icd_codes]
    if use_ov:
        columns += [x+'-OV' for x in acs_icd_codes]
    if use_death:
        columns += ['DEATH']

    assert time_span in ['30d']  # I guess we only have 30d outcome for now
    columns = [f'outcome-{time_span}-{x}' for x in columns]
    return Data(
        data[columns].any(axis=1).values,
        columns=[f'{time_span}_acs'],
    )


def make_comorbidity_features(data, combine_sv_ov=True):
    def remove_prefix_and_suffix(x):
        return x[7:-3]

    icd_codes = [
        remove_prefix_and_suffix(x)
        for x in data.filter(regex='prev.*SV').columns
    ]

    sv = data.filter(regex='prev.*SV').values
    ov = data.filter(regex='prev.*OV').values

    if combine_sv_ov:
        return Data(
            data=(sv | ov),
            columns=icd_codes,
        )

    else:
        return Data(
            data=np.concatenate([sv, ov], axis=1),
            columns=(
                [f"{x}_SV" for x in icd_codes] +
                [f"{x}_OV" for x in icd_codes]
            )
        )


def make_basic_features(data):
    data = data.loc[
           :, ['Vardkontakt_PatientAlderVidInskrivning', 'Patient_Kon']
           ].copy()
    data['male'] = (data.Patient_Kon == 'M').astype(int)
    data['female'] = (data.Patient_Kon == 'F').astype(int)

    return Data(
        data=data[[
            'Vardkontakt_PatientAlderVidInskrivning',
            'male', 'female',
        ]].values,
        columns=['age', 'male', 'female']
    )


class Base(Extractor):
    def get_data(self, brms) -> DataWrapper:
        raise NotImplementedError

    def get_labels(self, brsm):
        outcome_columns = ['I200', 'I21', 'I22', 'death']
        y = brsm.loc[:, outcome_columns].any(axis=1)
        return Data(y.values, columns=['ACS'])

    def _dev_test_split(self):
        # The idea is to use any events at Helsingborg as the test (held-out)
        # set. But some patients have been to multiple hospitals, including
        # Helsingborg - those visits should then be excluded.

        # Load the index data
        brsm = sk1718.load_outcomes_from_ab_brsm()

        # Create the whole dataset
        data = self.get_data(brsm)

        # Now do the hold-out split.
        # All events at Helsingborg...
        hbg = brsm.set_index('Alias').hospital == 'Helsingborgs lasarett'

        # All events for patients who at some point were at Helsingborg
        hbg_and_related = hbg.groupby('Alias').any().reindex(brsm.Alias)

        # All events not at Helsingborg, but where the patient has been
        # there at some point (These will be excluded)
        related = hbg ^ hbg_and_related

        # Each event is now mapped to 0 for development, 1 for test, and
        # -1 for hold-out.
        split = hbg.astype(int) + related.map({False: 0, True: -1})

        # Perform the actual split dev/test split
        dev_test_splitter = CrossValidationWrapper(
            PredefinedSplit(split.values)
        )
        return next(dev_test_splitter.split(data))

    def get_development_data(self) -> DataWrapper:
        dev, _ = self._dev_test_split()
        return dev

    def get_test_data(self) -> DataWrapper:
        _, test = self._dev_test_split()
        return test


class Flat(Base):
    def get_data(self, brsm) -> DataWrapper:
        patient_history = sk1718.summarize_patient_history(
            brsm, **self.features
        )

        # x_dict = {
        #     'basic': make_basic_features(brsm),
        #     'history': Data(
        #         patient_history.values,
        #         columns=list(patient_history.columns),
        #     )
        # }

        data = DataWrapper(
            features=Container({'history': Data(
                patient_history.values,
                columns=list(patient_history.columns),
            )}),
            labels=self.get_labels(brsm),
            index=Data(brsm.KontaktId.values, columns=['KontaktId']),
            groups=brsm.Alias.values,
            fits_in_memory=True
        )
        return data


class Ragged(Base):
    def get_test_data(self) -> DataWrapper:
        ...

    def get_development_data(self) -> DataWrapper:
        brsm = sk1718.load_outcomes_from_ab_brsm()
        mh = load('/mnt/air-crypt/air-crypt-esc-trop/axel/'
                  'sk1718_brsm_staggered_diagnoses.pickle')

        sv_cols = list(mh.filter(regex='_sv'))
        ov_cols = list(mh.filter(regex='_ov'))
        data = mh[sv_cols[:100] + ov_cols[:100]]

        row_starts = (
            mh.groupby('admission_index')[['data_index']]
            .first().join(brsm, how='outer')
            .fillna(method='bfill').data_index.astype(int)
        )
        row_ends = (
            mh.groupby('admission_index')[['data_index']]
            .last().join(brsm, how='outer')
            .fillna(method='ffill').data_index.astype(int) + 1
        )

        # x_dict = {}
        # x_dict['icd_history'] =

        # brsm, lab_values = load_and_clean_ab_brsm()
        #
        # # Always include age and sex
        # x_dict = {'basic': make_basic_features(brsm)}
        #
        # if 'lab_values' in self.features:
        #     x_dict['lab_values'] = Data(
        #         lab_values.values,
        #         columns=list(lab_values.columns)
        #     )
        #
        # if 'medicine' in self.features:
        #     x_dict['medicine'] = make_medicine_features(
        #         brsm, **self.features['medicine']
        #     )
        #
        # if 'comorbidities' in self.features:
        #     x_dict['comorbidities'] = make_comorbidity_features(
        #         brsm, **self.features['comorbidities']
        #     )
        y = brsm[['I200', 'I21', 'I22']].any(axis=1)
        data = DataWrapper(
            features=RaggedData(
                data.values,
                index=range(len(y)),
                slices=list(zip(row_starts, row_ends)),
                columns=list(data)
            ),
            labels=self.get_labels(brsm),
            index=Data(brsm.KontaktId.values, columns=['KontaktId']),
            # # labels=Data(y.values, columns=['ACS']),
            #
            #     # 'y': Data(y.values, columns=['ACS']),
            #     'index': Data(
            #         brsm.KontaktId.values,
            #         columns=['KontaktId']
            #     )
            # },
            # index=range(len(brsm)),
            groups=brsm.Alias.values,
            fits_in_memory=True
        )
        return data
        # return self.hold_out(data)
