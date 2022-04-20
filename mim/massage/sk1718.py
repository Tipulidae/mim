from os.path import join

import numpy as np
import pandas as pd

from mim.util.logs import get_logger
from mim.util.metadata import save

log = get_logger("Skåne 17-18 massage")


def read_csv(name, **kwargs):
    base_path = "/mnt/air-crypt/air-crypt-raw/andersb/data/Skane_17-18/" \
                "Uttag_1"
    return pd.read_csv(
        join(base_path, name),
        encoding='latin1',
        sep='|',
        **kwargs
    )


def last_in_clusters(seq, gap=30):
    # Given an input sequence of numbers, I want to return a
    # sorted sub-sequence such that each number is included at most once, and
    # the gap between the numbers is at least __gap__ large. Furthermore,
    # in a sub-sequence of numbers where the gap is smaller than __gap__,
    # I want to keep only the last number.

    # In other words, I want to first organize a sequence into clusters
    # based on their distances (gap), and then return the last number in each
    # cluster.
    if len(seq) == 0:
        return seq

    seq = sorted(set(seq))
    prev = seq[0]
    output = []
    for x in seq:
        if x - prev > gap:
            output.append(prev)

        prev = x
    output.append(seq[-1])
    return output


def make_chest_pain_index():
    liggaren = read_csv(
        'PATIENTLIGGAREN_Vårdkontakter_2017_2018.csv',
        parse_dates=[
            'Vardkontakt_InskrivningDatum',
            'Vardkontakt_UtskrivningDatum'
        ],
        usecols=[
            'KontaktId',
            'Alias',
            'BesokOrsak_Kod',
            'Vardkontakt_InskrivningDatum',
            'Vardkontakt_UtskrivningDatum'
        ]
    )
    # Take only those complaining about chest-pain
    brsm = liggaren[liggaren.BesokOrsak_Kod == 'BröstSm']
    # Drop duplicates
    brsm = brsm[~brsm.drop('KontaktId', axis=1).duplicated(keep='first')]
    return brsm.drop(columns=['BesokOrsak_Kod'])


def make_lab_values(index):
    lab = read_csv(
        'MELIOR_LabanalyserInom24TimmarFrånAnkomst.csv',
        parse_dates=['Analyssvar_ProvtagningDatum'],
        usecols=[
            'KontaktId',
            'Labanalys_Beskrivning',
            'Analyssvar_Varde',
            'Analyssvar_ProvtagningDatum',
        ]
    )

    lab = (
        index
        .loc[:, ['KontaktId', 'Vardkontakt_InskrivningDatum']]
        .join(lab.set_index('KontaktId'), on='KontaktId')
    )

    lab['minutes'] = (
        (lab.Analyssvar_ProvtagningDatum - lab.Vardkontakt_InskrivningDatum)
        .dt.total_seconds() // 60
    )
    return lab.drop(
        columns=[
            'Vardkontakt_InskrivningDatum',
            'Analyssvar_ProvtagningDatum'
        ]
    )


def massage_lab_values(lab):
    # Keep only the 50 most common lab values
    top50_lab_names = lab.Labanalys_Beskrivning.value_counts().index[:50]
    lab50 = lab.loc[lab.Labanalys_Beskrivning.isin(top50_lab_names), :]

    # Convert all the measurements to float (or nan), then drop nans
    lab50 = lab_values_to_float(lab50).dropna(how='any')
    return lab50


def lab_values_to_float(lab):
    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def lab_value_to_float(value):
        if is_float(value):
            return float(value)
        else:
            return coerce_lab_value(value)

    def coerce_lab_value(value):
        if value in inequality_to_float:
            return inequality_to_float[value]
        else:
            return np.nan

    inequality_to_float = {
        '<0.08': 0.07,
        '<2.0': 1.0,
        '>40': 41.0,
        '<0.0': -1.0,
        '<0.10': 0.09,
        '<0.1': 0.09,
        '<0.60': 0.59,
        '<50': 49.0,
        '>150': 151.0,
        '<20': 19.0,
        '<5': 4.0,
        '>9999': 10000.0,
        '<10': 9.0,
        '<0.01': 0.009,
        '>8.0': 9.0,
        '<3': 2.0,
        '<0.05': 0.04,
        '>10.0': 11.0,
        '<0.8': 0.7,
        '<0.02': 0.01,
        '>9998': 10000.0,
        '>31.0': 32.0,
    }
    lab = lab.copy()
    lab.Analyssvar_Varde = lab.Analyssvar_Varde.apply(lab_value_to_float)
    return lab


inequality_to_float = {
    '<0.08': 0.07,
    '<2.0': 1.0,
    '>40': 41.0,
    '<0.0': -1.0,
    '<0.10': 0.09,
    '<0.1': 0.09,
    '<0.60': 0.59,
    '<50': 49.0,
    '>150': 151.0,
    '<20': 19.0,
    '<5': 4.0,
    '>9999': 10000.0,
    '<10': 9.0,
    '<0.01': 0.009,
    '>8.0': 9.0,
    '<3': 2.0,
    '<0.05': 0.04,
    '>10.0': 11.0,
    '<0.8': 0.7,
    '<0.02': 0.01,
    '>9998': 10000.0,
    '>31.0': 32.0,
}


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def coerce_lab_value(value):
    if value in inequality_to_float:
        return inequality_to_float[value]
    else:
        return np.nan


def lab_value_to_float(value):
    if is_float(value):
        return float(value)
    else:
        return coerce_lab_value(value)


def calculate_time_chunks(lab):
    def discretize_time(df, gap=30):
        minutes = df.minutes.values
        return last_in_clusters(minutes, gap=gap)

    lab_times = (
        lab
        .groupby('KontaktId')
        .apply(discretize_time)
        .apply(pd.Series)
    )
    return lab_times


def slice_lab_values(index, lab, chunks, t_min=None, t_max=None):
    lab_names = list(sorted(lab.Labanalys_Beskrivning.unique()))

    lab_times = lab.join(chunks, on='KontaktId')
    if t_min is None:
        c1 = True
    else:
        c1 = lab_times.minutes > lab_times[t_min]

    if t_max is None:
        c2 = True
    else:
        c2 = lab_times.minutes <= lab_times[t_max]

    flat = (
        lab
        .loc[c1 & c2, :]
        .sort_values(by=['KontaktId', 'minutes'])
        .drop_duplicates(subset=['KontaktId', 'Labanalys_Beskrivning'],
                         keep='last')
        .set_index('KontaktId')
        .loc[:, ['Labanalys_Beskrivning', 'Analyssvar_Varde']]
    )
    data = pd.DataFrame(
        data={
            lab_name: flat.loc[
                flat.Labanalys_Beskrivning == lab_name, 'Analyssvar_Varde']
            for lab_name in lab_names
        },
        index=pd.Index(sorted(lab.KontaktId.unique()), name='KontaktId')
    )

    data_isna = data.isna().astype(int).rename(
        columns={x: x+'_isna' for x in data.columns}
    )

    result = pd.DataFrame(index=index.KontaktId)
    result = result.join(data).fillna(0)
    result = result.join(data_isna).fillna(1)
    return result


def make_data123(index, num_time_points=3):
    log.debug('Making lab-values')
    lab_values = make_lab_values(index)

    log.debug('Massaging lab-values')
    lab_values = massage_lab_values(lab_values)

    log.debug('Calculating time chunks')
    time_chunks = calculate_time_chunks(lab_values)

    log.debug('Slicing lab-values')
    ts = {
        't_min': [None, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        't_max': [0, 1, 2, 3, 4, 5, 6, 7, 8, None]
    }
    data = pd.concat(
        [
            slice_lab_values(
                index,
                lab_values,
                time_chunks,
                t_min=ts['t_min'][x],
                t_max=ts['t_max'][x]
            ) for x in range(num_time_points)
        ],
        keys=[f'data{x}' for x in range(1, num_time_points+1)],
        axis=1
    )

    return data


def make_pontus_lab_values(index):
    log.debug('Making lab-values')
    lab = make_lab_values(index)

    pontus_lab_values = [
        "P-Troponin T",
        "P-Kreatinin(enz)",
        "P-Glukos",
        "B-Hemoglobin(Hb)",
    ]
    lab = lab.loc[lab.Labanalys_Beskrivning.isin(pontus_lab_values), :]

    # Convert all the measurements to float (or nan), then drop nans
    lab = lab_values_to_float(lab).dropna(how='any')
    return lab


def load_outcomes_from_ab_brsm():
    colmap = {
        'Alias': 'Alias',
        'KontaktId': 'KontaktId',
        'LopNr': 'LopNr',
        'Vardkontakt_InskrivningDatum': 'admission_date',
        'Vardkontakt_PatientAlderVidInskrivning': 'age',
        'Patient_Kon': 'sex',
        'outcome-30d-I200-SV': 'I200',
        'outcome-30d-I21-SV': 'I21',
        'outcome-30d-I22-SV': 'I22',
        'outcome-30d-DEATH': 'death',
    }
    data = pd.read_csv(
        '/mnt/air-crypt/air-crypt-esc-trop/andersb/scratch/brsm-U.csv',
        parse_dates=['Vardkontakt_InskrivningDatum'],
        usecols=colmap.keys(),
    )

    return (
        data
        .rename(columns=colmap)
        .set_index(['Alias', 'admission_date'])
        .sort_index()
    )


def make_multihot_diagnoses(index):
    """

    :param index: DataFrame with columns LopNr and Alias.
    :return: DataFrame with Alias, diagnosis_date as multi-index, and
    around 1500 columns, one for each possible ICD10 diagnosis in the sos-
    material. Each row corresponds to a multi-hot encoding of a hospital
    visit. The columns are sorted by frequency, so that the first column is
    the most common diagnosis, and the last column is the least common.
    """
    # output: df with columns Alias, date, <icd1>, <icd2>, ...

    # assert any([sv, ov])

    diagnosis_cols = ['hdia'] + [f'DIA{x}' for x in range(1, 31)]

    def _multihot(path):
        log.info(f'Loading {path}')
        sos = read_csv(
            path,
            usecols=['LopNr', 'INDATUM'] + diagnosis_cols,
            parse_dates=['INDATUM'],
            low_memory=False,
        )
        log.info('Stacking diagnoses')
        sos = (
            sos
            .reset_index()  # We will use this to disambiguate same-day events
            .join(index.set_index('LopNr').Alias, on='LopNr', how='inner')
            .rename(columns={'INDATUM': 'diagnosis_date'})
            .set_index(['Alias', 'diagnosis_date', 'index'])
            .sort_index()
            .loc[:, diagnosis_cols]
            .stack()
            .rename('icd10')
            .reset_index(level=3, drop=True)
            .reset_index(level=[2, 1])
            .drop_duplicates()
        )

        log.info('Pivoting diagnoses')
        icd_order = (
            sos
            .icd10
            .value_counts()
            .reset_index()
            .sort_values(by=['icd10', 'index'], ascending=[False, True])
            .loc[:, 'index']
            .values
        )
        sos['diagnosis'] = True
        sos = sos.pivot_table(
            index=['Alias', 'diagnosis_date', 'index'],
            columns='icd10',
            values='diagnosis'
        )

        sos = sos.fillna(0).astype(bool).loc[:, icd_order]
        log.info('Done making multi-hot diagnoses!')
        return sos

    sv = _multihot('SOS_T_T_T_R_PAR_SV_24129_2020.csv')
    ov = _multihot('SOS_T_T_T_R_PAR_OV_24129_2020.csv')

    sv.columns += '_sv'
    ov.columns += '_ov'

    log.info('Concatenating diagnoses')
    svov = pd.concat([sv, ov], join='outer').fillna(False)
    return svov


def save_multihot_diagnoses():
    df = pd.read_csv(
        '/mnt/air-crypt/air-crypt-esc-trop/andersb/scratch/brsm-U.csv'
    )
    index = df[['LopNr', 'Alias', 'Vardkontakt_InskrivningDatum']]
    svov = make_multihot_diagnoses(index).astype(pd.SparseDtype(bool, False))
    save(svov, '/mnt/air-crypt/air-crypt-esc-trop/axel/'
               'sk1718_brsm_multihot.pickle')
