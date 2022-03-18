import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from mim.extractors.extractor import Data, Container, Extractor
from mim.cross_validation import CrossValidationWrapper
from mim.util.logs import get_logger
from mim.massage.sk1718 import lab_value_to_float

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
    lab_values = data.loc[:, lab_value_cols].applymap(lab_value_to_float)
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


class Sk1718(Extractor):
    def hold_out(self, data):
        if self.cv_kwargs is not None and 'test_size' in self.cv_kwargs:
            test_size = self.cv_kwargs['test_size']
        else:
            test_size = 1 / 4

        if test_size > 0:
            hold_out_splitter = CrossValidationWrapper(
                GroupShuffleSplit(
                    n_splits=1, test_size=test_size, random_state=42
                )
            )
            dev, _ = next(hold_out_splitter.split(data))

        else:
            dev = data

        return dev

    def get_data(self) -> Container:
        brsm, lab_values = load_and_clean_ab_brsm()

        # Always include age and sex
        x_dict = {'basic': make_basic_features(brsm)}

        if 'lab_values' in self.features:
            x_dict['lab_values'] = Data(
                lab_values.values,
                columns=list(lab_values.columns)
            )

        if 'medicine' in self.features:
            x_dict['medicine'] = make_medicine_features(
                brsm, **self.features['medicine']
            )

        if 'comorbidities' in self.features:
            x_dict['comorbidities'] = make_comorbidity_features(
                brsm, **self.features['comorbidities']
            )

        data = Container(
            {
                'x': Container(x_dict),
                'y': make_acs_labels(brsm, **self.labels),
                'index': Data(
                    brsm.KontaktId.values,
                    columns=['KontaktId']
                )
            },
            index=range(len(brsm)),
            groups=brsm.Alias.values,
            fits_in_memory=True
        )

        return self.hold_out(data)
