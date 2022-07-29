from os.path import join

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from mim.experiments.extractor import Data, RaggedData, Container, \
    Extractor, DataWrapper
from mim.cross_validation import CrossValidationWrapper
from mim.util.logs import get_logger
from mim.util.metadata import load
from massage import sk1718
# from massage import sk1718 as massage
# from massage.sk1718 import lab_value_to_float, load_outcomes_from_ab_brsm


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


# class NamesAreHard(Extractor):
#     def get_development_data(self) -> DataWrapper:
#         pass
#
#     def get_test_data(self) -> DataWrapper:
#         pass
#
#     def make_data(self):
#         base_path = '/mnt/air-crypt/air-crypt-esc-trop/axel/'
#         brsm = sk1718.load_outcomes_from_ab_brsm()
#         sv = load(join(base_path, 'sk1718_brsm_multihot_sv.pickle'))


# def load_multihot(brsm, sv=True, ov=True):
#     base_path = '/mnt/air-crypt/air-crypt-esc-trop/axel/'
#     if sv:
#          = load(join(base_path, 'sk1718_brsm_multihot_sv.pickle'))


class Base(Extractor):
    def get_data(self) -> DataWrapper:
        raise NotImplementedError

    def get_labels(self, brsm):
        outcome_columns = ['I200', 'I21', 'I22', 'death']
        y = brsm.loc[:, outcome_columns].any(axis=1)
        return Data(y.values, columns=['ACS'])

    def _dev_test_split(self):
        # Could parameterize the dev-test splitting, but I'm not sure I
        # actually want to. I want to use the same split each time.
        data = self.get_data()
        dev_test_splitter = CrossValidationWrapper(
            GroupShuffleSplit(
                n_splits=1, test_size=0.25, random_state=42
            )
        )
        return next(dev_test_splitter.split(data))

    def get_development_data(self) -> DataWrapper:
        dev, _ = self._dev_test_split()
        return dev

    def get_test_data(self) -> DataWrapper:
        _, test = self._dev_test_split()
        return test


def foobar(brsm, spec):
    # Trying to figure out the correct order to do things here...

    if 'sv' in spec['sources']:
        mh = sk1718._multihot('SOS_T_T_T_R_PAR_SV_24129_2020.csv', brsm)
        codes = []
        if 'diagnoses' in spec:
            k = spec['diagnoses']
            icd = mh.like(like='ICD_')
            icd = icd.iloc[:, :k]
            icd = sk1718.make_multihot_staggered(icd, brsm)
            icd = sk1718.sum_events_in_interval(
                icd, brsm, **spec['intervals']
            )
            # Columns will now be {I#}_{SV/OV}_{ICD/OP}_{xxx}
            # Index is the admission_index.
            icd.columns = [
                f"{interval}_SV_{code}" for interval, code in icd.columns
            ]
            codes.append(icd)
        if 'interventions' in spec:
            k = spec['interventions']
            op = mh.like(like='OP_')
            op = op.iloc[:, :k]
            op = sk1718.make_multihot_staggered(op, brsm)
            op = sk1718.sum_events_in_interval(
                op, brsm, **spec['intervals']
            )
            # Columns will now be {I#}_{SV/OV}_{ICD/OP}_{xxx}
            # Index is the admission_index.
            op.columns = [
                f"{interval}_SV_{code}" for interval, code in op.columns
            ]
            codes.append(op)

    brsm = sk1718.load_outcomes_from_ab_brsm()
    if 'sv' in spec['sources']:
        mh = sk1718._multihot("SOS_T_T_T_R_PAR_SV_24129_2020.csv", brsm)
        codes = []
        if 'diagnoses' in spec:
            codes.append(
                foo(source='sv', code='icd', k=spec['diagnoses'],
                    interval_spec=spec['intervals'], mh=mh, brsm=brsm)
            )
        if 'interventions' in spec:
            codes.append(
                foo(source='sv', code='op', k=spec['interventions'],
                    interval_spec=spec['intervals'], mh=mh, brsm=brsm)
            )

        # I think I need something like a map:
        bar = {'icd': 'diagnoses', 'op': 'interventions'}
        for code in ['icd', 'op']:
            if bar[code] in spec:
                codes.append(
                    foo(source='sv', code=code, k=spec[bar[code]],
                        interval_spec=spec['intervals'], mh=mh, brsm=brsm)
                )

    codes = []
    for source in ['SV', 'OV']:
        mh = sk1718._multihot(
            f"SOS_T_T_T_R_PAR_{source}_24129_2020.csv", brsm
        )
        bar = {'ICD': 'diagnoses', 'OP': 'interventions'}
        for code in ['ICD', 'OP']:
            if bar[code] in spec:
                codes.append(
                    foo(
                        source=source,
                        code=code,
                        k=spec[bar[code]],
                        interval_spec=spec['intervals'],
                        mh=mh,
                        brsm=brsm
                    )
                )
    return pd.concat(codes, axis=1)


def foo(source, code, k, interval_spec, mh, brsm):
    source = source.upper()
    code = code.upper()
    assert source in ['SV', 'OV']
    assert code in ['ICD', 'OP']

    mh = mh.filter(like=f"{code}_")
    mh = mh.iloc[:, k]
    mh = sk1718.make_multihot_staggered(mh, brsm)
    mh = sk1718.sum_events_in_interval(mh, brsm, **interval_spec)
    mh.columns = [
        f"{interval}_{source}_{code}" for interval, code in mh.columns
    ]

    # Columns will now be {I#}_{SV/OV}_{ICD/OP}_{xxx}
    # Index is the admission_index.
    return mh


class Flat(Base):
    def get_data(self) -> DataWrapper:
        brsm = sk1718.load_outcomes_from_ab_brsm()
        # sv = sk1718._multihot('SOS_T_T_T_R_PAR_SV_24129_2020.csv', brsm)

        base_path = '/mnt/air-crypt/air-crypt-esc-trop/axel/'

        # if 'sv' in self.features:
        # sv = sk1718._multihot('SOS_T_T_T_R_PAR_SV_24129_2020.csv', brsm)
        # sv = load(join(base_path, 'sk1718_brsm_multihot_sv.pickle'))
        # sv = load(
        #     join(base_path, 'sk1718_brsm_multihot_sv.pickle'),
        #     allow_uncommitted=True
        # )
        sv = load(
            join(base_path, 'sk1718_brsm_staggered_diagnoses.pickle'),
            allow_uncommitted=True
        )
        k = 10

        # levels will be:
        # Interval, SV/OV, ICD/OP,
        sv_icd = sk1718.sum_events_in_interval(
            sv['ICD'].iloc[:, :k],
            brsm,
            periods=5
        )
        sv_op = sk1718.sum_events_in_interval(
            sv['OP'].iloc[:, :k],
            brsm,
            periods=5
        )

        sv = pd.concat([sv_icd, sv_op], axis=1, keys=['ICD', 'OP'])

        data = DataWrapper(
            features=Data(
                sv.values,
                columns=list(map('_'.join, sv.columns)),
            ),
            labels=self.get_labels(brsm),
            index=Data(brsm.KontaktId.values, columns=['KontaktId']),
            groups=brsm.Alias.values,
            fits_in_memory=True
        )
        return data


class Ragged(Base):
    def get_development_data(self) -> Container:
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

        data = DataWrapper(
            features=RaggedData(
                data.values,
                # index=range(len(y)),
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

        return self.hold_out(data)
