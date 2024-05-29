from os.path import join

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from massage.sos_util import fix_dors_date
from massage.carlson_ecg import (
    ECGStatus,
    expected_lead_names,
    glasgow_vector_names,
    glasgow_scalar_names,
    glasgow_diagnoses,
    glasgow_rhythms
)
from mim.cache.decorator import cache
from mim.util.logs import get_logger


log = get_logger("ESC-Trop Massage")


ECG_PATH = '/projects/air-crypt/legacy/air-crypt-esc-trop/axel/ecg.hdf5'

important_status_labels = {
    ECGStatus.MISSING_DATA,
    ECGStatus.MISSING_LABELS,
    ECGStatus.MISSING_ECG,
    ECGStatus.MISSING_BEAT,
    ECGStatus.MISSING_RHYTHM,
    ECGStatus.TECHNICAL_ERROR,
    ECGStatus.BAD_DIAGNOSIS,
    ECGStatus.FILE_MISSING,
    ECGStatus.BAD_ECG_DIMENSIONS,
    ECGStatus.BAD_BEAT_DIMENSIONS,
    ECGStatus.EMPTY_ECG_ROWS,
    ECGStatus.EMPTY_ECG_COLUMNS,
    ECGStatus.EMPTY_BEAT_ROWS,
    ECGStatus.EMPTY_BEAT_COLUMNS,
    ECGStatus.MISSING_RECORDING,
    ECGStatus.MISSING_FILE_FORMAT,
    ECGStatus.MISSING_ID,
    ECGStatus.MISSING_DATE,
    ECGStatus.BAD_DATE,
    ECGStatus.MISSING_PATIENT,
    ECGStatus.BAD_LEAD_NAMES
}

mace_codes_new = [
    "I200",
    "I21",
    "I22",
    "I441",
    "I442",
    "I46",
    "I470",
    "I472",
    "I490",
    "J819",
    "R570"
]

mace_codes_paul = [
    "I200",
    "I21",
    "I220",
    "I221",
    "I228",
    "I229",
    "I249",
    "I441",
    "I442",
    "I460",
    "I461",
    "I469",
    "I472",
    "I490",
    "I500",
    "I501",
    "I509",
    "J819",
    "R570",
    "R578",
    "R579",
    "R960",
    "R961",
    "R989",
    "R999",
]

mace_codes_paul_new = [
    "DF005",
    "DF017",
    "DF025",
    "DF028",
    "DG017",
    "DG021",
    "DG023",
    "DG026",
    "FNA00",
    "FNA10",
    "FNC10",
    "FNC20",
    "FNC30",
    "FNF96",
    "FNG00",
    "FNG02",
    "FNG05",
    "FPE00",
    "FPE26",
    "FPE20",
    "FPE10",
    "TFP00",
    "I200",
    "I210",
    "I211",
    "I212",
    "I213",
    "I214",
    "I214A",
    "I214B",
    "I214W",
    "I214X",
    "I219",
    "I220",
    "I228",
    "I229",
    "I249",
    "I441",
    "I441B",
    "I442",
    "I460",
    "I469",
    "I472",
    "I472A",
    "I472B",
    "I472C",
    "I490",
    "I500",
    "I501",
    "I509",
    "J819",
    "R570",
    "R579",
]

mace_codes_anders = [
    "I200",
    "I21-P",
    "I210",
    "I211",
    "I212",
    "I213",
    "I214",
    "I214A",
    "I214B",
    "I214W",
    "I214X",
    "I219",
    "I220",
    "I221",
    "I228",
    "I229",
    "I441B",
    "I442",
    "I46-",
    "I460",
    "I461",
    "I469",
    "I472",
    "I472A",
    "I472B",
    "I472C",
    "I490",
    "R960",
    "R961",
    "R989",
    "R99-P",
    "R999",
    "R570"
]


def make_ecg_table():
    with h5py.File(ECG_PATH, 'r') as ecg:
        table = pd.DataFrame(
            pd.to_datetime(ecg['meta']['date'][:].astype(str)),
            columns=['ecg_date']
        )
        table['Alias'] = '{' + pd.Series(
            ecg['meta']['alias'][:].astype(str)
        ).str.upper() + '}'

        status = pd.DataFrame(
            ecg['meta']['status'][:],
            columns=ecg['meta']['status_keys'][:].astype(str)
        )
        status['USABLE'] = ~status[list(map(
            lambda s: s.name, important_status_labels))].any(axis=1)

        table = table.loc[status.USABLE & (table.ecg_date >= '1999')]
        return table


def _make_double_ecg(index, min_age_seconds=-3600, max_age_seconds=7200):
    ecg = (
        make_ecg_table()
        .rename_axis('ecg')
        .reset_index()
        .set_index('Alias')
    )

    ecg = index.join(ecg, how='left').reset_index()
    ecg = ecg.sort_values(by=['Alias', 'ecg_date'])
    dt = (ecg.ecg_date - ecg.admission_date).dt.total_seconds()
    ecg = ecg[['Alias', 'ecg', 'ecg_date']]

    before = ecg.loc[(dt > min_age_seconds) & (dt < 0), :].drop_duplicates(
        subset=['Alias'], keep='last')
    after = ecg.loc[(dt >= 0) & (dt < max_age_seconds), :].drop_duplicates(
        subset=['Alias'], keep='first')

    ecg_0 = (
        pd.concat([before, after], axis=0)
        .sort_values(by=['Alias', 'ecg_date'])
        .drop_duplicates(subset=['Alias'], keep='last')
        .set_index('Alias')
    )

    ecg_1 = (
        ecg.loc[(dt < -7 * 24 * 3600), :]
        .drop_duplicates(subset=['Alias'], keep='last')
        .set_index('Alias')
    )
    return ecg_0.join(ecg_1, how='outer', lsuffix='_0', rsuffix='_1')


def make_ed_ecgs(index, min_age_seconds=-3600, max_age_seconds=7200,
                 pivot=False):
    ecg = (
        make_ecg_table()
        .rename_axis('ecg')
        .reset_index()
        .set_index('Alias')
    )
    ecg = index.join(ecg, how='inner').reset_index()

    dt = (ecg.ecg_date - ecg.admission_date).dt.total_seconds()
    ecg.loc[:, 'dt'] = dt
    ecg.loc[:, 'abs_dt'] = dt.abs()
    ecg = ecg.loc[(dt >= min_age_seconds) & (dt < max_age_seconds), :]

    # I want to use the ECGs taken at the ED first, and only consider
    # pre-hospital ECGs after that.
    ecg.loc[:, 'priority'] = ecg.dt
    ecg.loc[ecg.dt < 0, 'priority'] = (
            max_age_seconds - ecg.loc[ecg.dt < 0, 'priority'])

    ecg = ecg.sort_values(by=['Alias', 'priority'])
    ecg.loc[:, 'n'] = 1
    ecg.loc[:, 'n'] = ecg.groupby('id')['n'].cumsum().astype(str)

    # I can now pivot to get a table with one column per ECG:
    if pivot:
        ecg8 = ecg.reset_index()[['Alias', 'ecg', 'dt', 'n']].pivot(
            index='Alias', columns='n', values=['ecg', 'dt'])
        ecg8.columns = ecg8.columns.map('_'.join)
        return ecg8

    return ecg


def make_forberg_features(ecg_ids):
    ecg_path = '/projects/air-crypt/legacy/air-crypt-esc-trop/axel/ecg.hdf5'
    with h5py.File(ecg_path, 'r') as ecg:
        glasgow_features = []
        for x in tqdm(ecg_ids, desc='Extracting Glasgow vectors'):
            glasgow_features.append(ecg['glasgow']['vectors'][x])

        vector_names = list(ecg['meta']['glasgow_vector_names'][:].astype(str))
        lead_names = ecg['meta']['lead_names'][:].astype(str)

    cols = [
        'Qamplitude',
        'Ramplitude',
        'Samplitude',
        'Tpos_amp',
        'Tneg_amp',
        'Qduration',
        'QRSduration',
        'Rduration',
        'Sduration',
        'QRSarea',
        'STslope',
        'ST_amp',
        'STT28_amp',
        'STT38_amp',
    ]
    index = [vector_names.index(name) for name in cols]

    glasgow_features = np.stack(glasgow_features)[:, index, :]
    final_five_positive = np.maximum(glasgow_features[:, -5:, :], 0)
    final_five_negative = np.maximum(-glasgow_features[:, -5:, :], 0)

    glasgow_features[:, -5:, :] = final_five_positive
    glasgow_features = np.append(glasgow_features, final_five_negative, axis=1)
    glasgow_features = glasgow_features.reshape((len(ecg_ids), -1))

    new_cols = (
        cols[:-5] +
        [f"{col}_{sign}" for sign in ['pos', 'neg'] for col in cols[-5:]]
    )

    columns = [f"{col}_{lead}" for col in new_cols for lead in lead_names]
    df = pd.DataFrame(glasgow_features, index=ecg_ids.index, columns=columns)
    return df


@cache
def make_johansson_features(ecg_ids):
    ecg_path = '/projects/air-crypt/legacy/air-crypt-esc-trop/axel/ecg.hdf5'
    with h5py.File(ecg_path, 'r') as ecg:
        glasgow_features = []
        glasgow_scalars = []
        for x in tqdm(ecg_ids, desc='Extracting Glasgow vectors'):
            glasgow_features.append(ecg['glasgow']['vectors'][x])
            glasgow_scalars.append(ecg['glasgow']['scalars'][x])

        vector_names = list(ecg['meta']['glasgow_vector_names'][:].astype(str))
        scalar_names = list(ecg['meta']['glasgow_scalar_names'][:].astype(str))
        lead_names = ecg['meta']['lead_names'][:].astype(str)

    vector_cols = [
        'EndQRSnotch_amp',
        'QRSarea',
        'Qamplitude',
        'Ramplitude',
        'ST60_amp',
        'ST80amplitude',
        'STM_amp',
        'STT28_amp',
        'STT38_amp',
        'STTmid_amp',
        'ST_amp',
        'Samplitude',
        'Tarea',
        'Tduration',
        'Tneg_amp',
        'Tpos_amp',
        'Tpos_dur'
    ]
    vector_index = [vector_names.index(name) for name in vector_cols]

    # Mapping -32768 to 0 for EndQRSnotch_amp
    glasgow_features = np.stack(glasgow_features)[:, vector_index, :]
    glasgow_features[:, 0, :] = np.where(
        glasgow_features[:, 0, :] == -32768.0,
        np.zeros_like(glasgow_features[:, 0, :]),
        glasgow_features[:, 0, :],
    )

    # New feature: qrst-fraction := qrs area divided by t area
    qrs_area = glasgow_features[:, 1, :]
    t_area = glasgow_features[:, 12, :]
    qrst_fraction = np.divide(
        qrs_area, t_area, out=np.ones_like(qrs_area), where=t_area != 0.0)
    vector_cols.append('QRST_fraction')
    glasgow_features = np.append(
        glasgow_features,
        np.expand_dims(qrst_fraction, axis=1),
        axis=1
    )

    # Flatten all the vector-valued glasgow features
    glasgow_features = glasgow_features.reshape((len(ecg_ids), -1))
    columns = [f"{col}_{lead}" for col in vector_cols for lead in lead_names]

    # Glasgow scalar variables
    scalar_cols = [
        'HeartRate',
        'HeartRateVariability',
        'LVstrain'
    ]
    scalar_index = [scalar_names.index(name) for name in scalar_cols]
    glasgow_scalars = np.stack(glasgow_scalars)[:, scalar_index]

    # Map -32768 to 2 for LVstrain
    glasgow_scalars[:, 2] = np.where(
        glasgow_scalars[:, 2] == -32768.0,
        2 * np.ones_like(glasgow_scalars[:, 2]),
        glasgow_scalars[:, 2],
    )

    # Combine scalar and vector features
    glasgow_features = np.concatenate(
        (glasgow_features, glasgow_scalars), axis=1)
    columns.extend(scalar_cols)

    # Build dataframe and return
    df = pd.DataFrame(glasgow_features, index=ecg_ids.index, columns=columns)
    return df


def make_ed_features(index):
    ed = read_csv(
        "ESC_TROP_Vårdkontakt_InkluderadeIndexBesök_2017_2018.csv"
    ).set_index('Alias')
    col_map = {
        'KontaktId': 'id',
        'Kön': 'male',
        'Ålder vid inklusion': 'age',
    }
    ed = ed.loc[:, list(col_map)].rename(columns=col_map)
    ed = ed[ed.id.isin(index.id)]

    ed.male = ed.male.apply(lambda x: 1 if x == 'M' else 0)

    return ed.drop(columns=['id']).loc[index.index, :]


def make_lab_features(index):
    tnt = make_troponin_table()
    tnt = index.reset_index().set_index('id').join(tnt, how='left')
    return tnt.set_index('Alias').drop(columns=['admission_date'])


def make_troponin_table():
    tnt = load_tnt()

    first_tnt = tnt.drop_duplicates(subset=['id'], keep='first')
    second_tnt = tnt[~tnt.index.isin(first_tnt.index)].drop_duplicates(
        subset=['id'], keep='first')

    r = first_tnt.set_index('id').join(
            second_tnt.set_index('id'),
            lsuffix='_1',
            rsuffix='_2'
        )

    r["log_tnt_1"] = np.log2(r["tnt_1"])
    r["log_tnt_2"] = np.log2(r["tnt_2"])

    return r


def make_double_ecg_features(index, include_delta_t=False):
    ecg = _make_double_ecg(index)
    ecg = index.join(ecg)
    ecg['delta_t'] = (ecg.admission_date - ecg.ecg_date_1).dt.total_seconds()
    ecg.delta_t /= 24 * 3600
    ecg['log_dt'] = (np.log10(ecg.delta_t) - 2.5) / 2  # Normalizing

    features = ['ecg_0', 'ecg_1', 'log_dt']
    if include_delta_t:
        features.append('delta_t')
    return ecg[features]


def read_csv(name, **kwargs):
    base_path = "/projects/air-crypt/air-crypt-raw/andersb/data/" \
                "ESC_Trop_17-18-2020-09-21/data"
    return pd.read_csv(
        join(base_path, name),
        encoding='latin1',
        sep='|',
        **kwargs
    )


def make_pauls_mace(index_visits):
    mace = read_csv(
        "ESC_TROP_Vårdkontakt_InkluderadeIndexBesök_2017_2018.csv",
        usecols=['MACE inom 30 dagar', 'KontaktId']
    )
    mace = mace.rename(
        columns={'MACE inom 30 dagar': 'mace30', 'KontaktId': 'id'}
    ).set_index('id')

    return (
        index_visits
        .reset_index()
        .set_index('id')
        .join(mace)
        .reset_index()
        .set_index('Alias')
        .loc[:, ['mace30']]
    )


def make_mace_table(index_visits, include_interventions=True,
                    include_deaths=True, icd_definition='new',
                    use_melior=False, use_sos=True, use_hia=True,
                    use_paul=False):
    if use_paul:
        return make_pauls_mace(index_visits)

    sources_of_mace = []

    if icd_definition == 'paul':
        icds_defining_mace = mace_codes_paul
    elif icd_definition == 'paul_new':
        icds_defining_mace = mace_codes_paul_new
    elif icd_definition == 'anders':
        icds_defining_mace = mace_codes_anders
    else:
        icds_defining_mace = mace_codes_new

    sources_of_mace.append(
        _make_mace_diagnoses(
            index_visits,
            icds_defining_mace=icds_defining_mace,
            use_melior=use_melior,
            use_sos=use_sos,
            use_hia=use_hia,
        )
    )

    if include_interventions:
        sources_of_mace.append(_make_mace_interventions(index_visits))
    if include_deaths:
        sources_of_mace.append(_make_mace_deaths(index_visits))
    if len(sources_of_mace) > 1:
        mace = pd.concat(sources_of_mace, axis=1)
    else:
        mace = sources_of_mace[0]

    # mace['mace30'] = mace.any(axis=1)
    # mace['ami30'] = mace[['I21', 'I22']].any(axis=1)
    return mace


def make_mace30_dataframe(mace_table):
    return pd.DataFrame(
        data=mace_table.any(axis=1).astype(int),
        index=mace_table.index,
        columns=['mace30']
    )


def make_ami30_dataframe(mace_table):
    return pd.DataFrame(
        data=mace_table[['I21', 'I22']].any(axis=1).astype(int),
        index=mace_table.index,
        columns=['ami30']
    )


def make_mace_chapters_dataframe(mace_table):
    # I decided to exclude a few of the chapters here because they are so
    # rare that they won't show up in the validation set.
    # I22 occurs for 4 patients in total, I put it together with I21.
    # R57 occurs for 3 patients in total, I remove it.
    # TF only happens for a single patient, I remove it.
    # DF is only 10 patients, I remove that as well.
    # I leave I49, even though it is only 12 patients. It's possible, maybe
    # even likely, that we're missing a few actual I49, if they are perhaps
    # entered into melior but not sos or swedeheart.
    chapters = [
        'I20', 'I21', 'I22', 'I44', 'I46', 'I47', 'I49', 'J81', 'R57',
        'DF', 'FN', 'FP', 'TF', 'death'
    ]
    df = pd.concat(
        [mace_table.filter(like=chapter).any(axis=1).rename(chapter) for
         chapter in chapters],
        axis=1
    )
    df['I21_22'] = df[['I21', 'I22']].any(axis=1)
    df = df.drop(columns=['I21', 'I22', 'R57', 'TF', 'DF'])
    return df.astype(int)


def _make_diagnoses_from_sos():
    diagnosis_cols = ['hdia'] + [f'DIA{x}' for x in range(1, 31)]

    sos_sv = pd.read_csv(
        '/projects/air-crypt/air-crypt-raw/andersb/data/'
        'Socialstyrelsen-2020-03-10/csv_files/sv_esctrop.csv',
        usecols=['Alias', 'INDATUM'] + diagnosis_cols,
        dtype=str
    )
    sos_sv = sos_sv.dropna(subset=['Alias']).set_index('Alias')
    sos_sv['diagnosis_date'] = pd.to_datetime(sos_sv.INDATUM)

    diagnoses = (
        sos_sv.set_index('diagnosis_date', append=True)
        .loc[:, diagnosis_cols]
        .stack()
        .rename('icd10')
        .reset_index(level=2, drop=True)
        .reset_index(level=1)
    )
    return diagnoses


def _make_diagnoses_from_rikshia():
    rikshia = read_csv(
        'ESC_TROP_SWEDEHEART_DAT221_rikshia_pop1.csv',
        dtype=str
    )

    rikshia.ADMISSION_ER_DATE = rikshia.ADMISSION_ER_DATE.fillna(
        rikshia.ADMISSION_DATE)
    rikshia.ADMISSION_ER_TIME = rikshia.ADMISSION_ER_TIME.fillna(
        rikshia.ADMISSION_TIME)

    rikshia['diagnosis_date'] = pd.to_datetime(
        rikshia.ADMISSION_ER_DATE + ' ' + rikshia.ADMISSION_ER_TIME)
    rikshia = rikshia.set_index('Alias')

    diagnoses = (
        rikshia.set_index('diagnosis_date', append=True)
        .filter(regex='(diag[0-9]+)', axis=1)
        .stack()
        .rename('icd10')
        .reset_index(level=2, drop=True)
        .reset_index(level=1)
    )

    return diagnoses


def remove_diagnoses_outside_time_interval(
        diags, index_visits, interval_days_start=0, interval_days_end=30):
    # diags is a DataFrame with index Alias and columns
    # "icd10" and "diagnosis_date". There is one row for each diagnosis,
    # so one patient can have multiple rows.
    diags = diags.join(index_visits)

    # Calculate the time between admission and diagnosis, in days
    dt = (diags.diagnosis_date.dt.floor('1D') -
          diags.admission_date.dt.floor('1D'))
    dt = dt.dt.total_seconds() / (3600 * 24)

    # Keep only the rows within the specified interval.
    return diags[(dt >= interval_days_start) & (dt <= interval_days_end)]


def _make_mace_diagnoses(
        index_visits, icds_defining_mace=None, use_melior=True, use_sos=True,
        use_hia=True, use_dors=True):
    if icds_defining_mace is None:
        icds_defining_mace = mace_codes_new

    assert any([use_melior, use_sos, use_hia])

    diagnose_sources = []

    if use_melior:
        during = _make_diagnoses(
            "ESC_TROP_Diagnoser_InkluderadeIndexBesök_2017_2018.csv")
        after = _make_diagnoses(
           "ESC_TROP_Diagnoser_EfterInkluderadeIndexBesök_2017_2018.csv")
        diagnose_sources.append(
            remove_diagnoses_outside_time_interval(during, index_visits))
        diagnose_sources.append(
            remove_diagnoses_outside_time_interval(after, index_visits))

    if use_sos:
        diagnose_sources.append(
            remove_diagnoses_outside_time_interval(
                _make_diagnoses_from_sos(), index_visits)
        )

    if use_hia:
        diagnose_sources.append(
            remove_diagnoses_outside_time_interval(
                _make_diagnoses_from_rikshia(),
                index_visits
            )
        )
    if use_dors:
        # When the day of death is unknown, we have set it to be the first of
        # the month. For this reason, we want to include deaths from a month
        # prior to the index as well (the patient is unlikely to go to the ED
        # after dying, so it should be all right), so that if the patient
        # arrived at the ED in the middle of the month and then died that
        # month at some unknown day, it still counts as having occurred within
        # 30 days of the index.
        diagnose_sources.append(
            remove_diagnoses_outside_time_interval(
                _make_diagnoses_from_dors(),
                index_visits,
                interval_days_start=-30,
                interval_days_end=30
            )
        )

    diagnoses = pd.concat(diagnose_sources, axis=0)

    # Create one column of bools per ICD-code included in our definition of
    # MACE. The table now contains one row per diagnosis, and one column per
    # ICD-code we are interested in.
    for icd in icds_defining_mace:
        diagnoses[icd] = diagnoses.icd10.str.contains(icd)

    # Our final table has one row per patient and one column of bools per
    # ICD-code in MACE.
    mace_icd10 = pd.DataFrame(index=index_visits.index)
    mace_icd10 = mace_icd10.join(
        diagnoses.groupby('Alias')[icds_defining_mace].any(),
        how='left'
    )
    return mace_icd10.fillna(False)


def make_index_visits(
        exclude_stemi=True,
        exclude_missing_tnt=True,
        exclude_missing_ecg=True,
        exclude_missing_old_ecg=True,
        exclude_missing_chest_pain=True,
        exclude_non_swedish_patients=True,
):
    """
    Creates a DataFrame with Alias as index and columns, admission_date and
    id, which corresponds to KontaktId in the original CSV-file.
    """
    index_visits = read_csv(
        "ESC_TROP_Vårdkontakt_InkluderadeIndexBesök_2017_2018.csv"
    )
    n = len(index_visits)
    log.debug(f'{n} index visits loaded from file.')

    if exclude_missing_chest_pain:
        index_visits = index_visits.dropna(subset=['BesokOrsakId'])
        n = len(index_visits)
        log.debug(f'{n} patients with chest pain')

    index_visits = index_visits[[
        'Alias', 'Vardkontakt_InskrivningDatum', 'KontaktId']]
    index_visits = index_visits.rename(
        columns={
            'Vardkontakt_InskrivningDatum': 'admission_date',
            'KontaktId': 'id'
        }
    )
    index_visits.admission_date = pd.to_datetime(index_visits.admission_date)
    index_visits = (
        index_visits
        .sort_values(by=['admission_date', 'id'])
        .drop_duplicates(subset=['Alias'], keep='first')
        .set_index('Alias')
    )
    n = len(index_visits)
    log.debug(f'{n} unique patients.')

    if exclude_non_swedish_patients:
        non_swedish_patients = [
            '{5D0F827F-BA44-45D1-A769-B025967BD156}',
            '{3C6AE413-FE02-49D9-8EBF-6807EA20D156}'
        ]
        index_visits = index_visits.loc[
            index_visits.index.difference(non_swedish_patients)
        ]
        n = len(index_visits)
        log.debug(f"{n} patients after excluding non-swedish patients")

    if exclude_stemi:
        stemi = index_visits.join(_make_index_stemi())
        dt = (stemi.admission_date - stemi.stemi_date
              ).dt.total_seconds() / (24*3600)
        # [-48, 24] hour interval after discussion with Jenny
        index_stemi = stemi[(dt >= -2) & (dt <= 1)].index.unique()
        index_visits = index_visits[~index_visits.index.isin(index_stemi)]
        n = len(index_visits)
        log.debug(f'{n} patients after excluding index STEMI')

    if exclude_missing_tnt:
        tnt = make_troponin_table()
        index_visits = index_visits[~index_visits.id.map(tnt.tnt_1).isna()]
        n = len(index_visits)
        log.debug(f'{n} patients after excluding missing index TnT')

    if exclude_missing_ecg or exclude_missing_old_ecg:
        ecg = _make_double_ecg(index_visits)
        if exclude_missing_ecg:
            index_visits = index_visits[index_visits.index.isin(
                ecg.dropna(subset=['ecg_0']).index)]
            n = len(index_visits)
            log.debug(f'{n} patients after excluding missing index ECG')

        if exclude_missing_old_ecg:
            index_visits = index_visits[index_visits.index.isin(
                ecg.dropna(subset=['ecg_1']).index)]
            n = len(index_visits)
            log.debug(f'{n} patients after excluding missing old ECG')

    index_visits = index_visits.sort_values(by=['admission_date', 'id'])
    assert index_visits.admission_date.is_monotonic_increasing
    return index_visits


def _make_mace_interventions(index_visits):
    actions_current = _make_actions(
        "ESC_TROP_PatientÅtgärder_InkluderadeIndexBesök_2017_2018.csv"
    )
    actions_after = _make_actions(
        "ESC_TROP_PatientÅtgärder_Efter_InkluderadeIndexBesök_2017_2018.csv"
    )

    actions = pd.concat([
        actions_current,
        actions_after
    ], axis=0)

    actions = actions.join(index_visits)

    dt = actions['date'] - actions.admission_date
    dt = dt.dt.total_seconds() / (3600 * 24)

    actions = actions[(dt >= 0) & (dt <= 30)]

    mace_kva = [
        "DF005",
        "DF017",
        "DF025",
        "DF028",
        "FNA00",
        "FNA10",
        "FNC10",
        "FNC20",
        "FNC30",
        "FNF96",
        "FNG00",
        "FNG02",
        "FNG05",
        "FPE00",
        "FPE26",
        "FPE20",
        "FPE10",
        "TFP00",
    ]
    for action in mace_kva:
        actions[action] = actions.action.str.contains(action)

    mace_actions = pd.DataFrame(index=index_visits.index)
    mace_actions = mace_actions.join(
        actions.groupby('Alias')[mace_kva].any(),
        how='left'
    )
    return mace_actions.fillna(False)


def _make_actions(csv_name):
    actions = read_csv(
        csv_name,
        usecols=[
            "Alias", "PatientAtgard_Kod", "PatientAtgard_ModifieradDatum"
        ]
    )

    actions = actions.rename(
        columns={
            'PatientAtgard_Kod': 'action',
            'PatientAtgard_ModifieradDatum': 'date'
        }
    )
    actions['date'] = pd.to_datetime(actions['date'])
    return actions.set_index('Alias')


def _make_diagnoses(csv_name):
    diagnoses = read_csv(
        csv_name,
        usecols=[
            'Alias', 'PatientDiagnos_Kod', 'PatientDiagnos_ModifieradDatum'
        ]
    )

    diagnoses = diagnoses.rename(
        columns={
            'PatientDiagnos_Kod': 'icd10',
            'PatientDiagnos_ModifieradDatum': 'diagnosis_date'
        }
    )
    diagnoses.diagnosis_date = pd.to_datetime(diagnoses.diagnosis_date)
    return diagnoses.set_index('Alias')


def _make_mace_deaths(index_visits):
    deaths = read_csv('ESC_TROP_SOS_R_DORS__14204_2019.csv')
    deaths = deaths.set_index('Alias')[['DODSDAT']].rename(
        columns={'DODSDAT': 'diagnosis_date'})

    deaths.diagnosis_date = pd.to_datetime(
        deaths.diagnosis_date.astype(str).apply(fix_dors_date),
        format='%Y%m%d'
    )
    deaths = index_visits.join(deaths)
    dt = (deaths.diagnosis_date -
          deaths.admission_date).dt.total_seconds() / (3600 * 24)
    deaths['death'] = (dt >= -30) & (dt <= 30)
    return deaths[['death']]


def _make_diagnoses_from_dors(include_secondary=True):
    deaths = read_csv('ESC_TROP_SOS_R_DORS__14204_2019.csv')
    causes = '(ULORSAK)'
    if include_secondary:
        causes += '|(MORSAK)'

    diagnoses = (
        deaths
        .set_index('Alias')
        .filter(regex=causes, axis=1)
        .stack()
        .reset_index(level=1, drop=True)
    )
    diagnoses = pd.DataFrame(diagnoses.rename('icd10'))
    diagnoses = diagnoses.join(
        deaths.set_index('Alias')[['DODSDAT']],
        how='left'
    )
    diagnoses = diagnoses.rename(columns={'DODSDAT': 'diagnosis_date'})
    diagnoses['diagnosis_date'] = pd.to_datetime(
        diagnoses.diagnosis_date.astype(str).apply(fix_dors_date),
        format='%Y%m%d',
    )
    return diagnoses.dropna()


def _make_index_stemi():
    hia = read_csv(
        'ESC_TROP_SWEDEHEART_DAT221_rikshia_pop1.csv',
        usecols=[
            'Alias',
            'ECG_STT_CHANGES',
            'INFARCTTYPE',
            'ADMISSION_ER_DATE',
            'ADMISSION_ER_TIME',
            'ADMISSION_DATE',
            'ADMISSION_TIME'
        ]
    )

    hia.ADMISSION_ER_DATE = hia.ADMISSION_ER_DATE.fillna(hia.ADMISSION_DATE)
    hia.ADMISSION_ER_TIME = hia.ADMISSION_ER_TIME.fillna(hia.ADMISSION_TIME)

    hia['stemi_date'] = pd.to_datetime(
        hia.ADMISSION_ER_DATE + ' ' + hia.ADMISSION_ER_TIME)
    hia = hia.set_index('Alias')

    return hia.loc[hia.INFARCTTYPE == 'STEMI', 'stemi_date']


def make_dataframe_for_anders():
    """
    Creates a dataframe with all the necessary data for Anders to test the
    models for the 1-TnT article by Pontus.

    The dataframe has Alias as index, and contains the four lab-values
    creatinine, glucose, hemoglobin and troponin. It also contains age, sex,
    path to the ECG matlab file, admission date and the target (AMI or death
    within 30 days, according to either Melior or SOS/HIA).

    The lab-values uses the first valid index-measurement for each patient,
    excluding values that are more than four hours after admission. Troponin
    measurements marked as '<5' are replaced with 4. All other non-numeric
    lab-values are excluded.

    The ECG used is the closest in time to admission, in the interval [-1, 4]
    hours.

    Only patients with valid entries for all columns are used.
    """
    # Create the index, to associate KontaktId with Alias
    # And also grab admission date, age and sex while we're at it
    log.debug("Reading from 'liggaren'")
    index_colmap = {
        'KontaktId': 'ed_id',
        'Alias': 'Alias',
        'Vardkontakt_InskrivningDatum': 'admission_date',
        'Kön': 'male',
        'Ålder vid inklusion': 'age',
        'BesokOrsakId': 'cause',
        'Sjukhus_Namn': 'hospital',
    }

    index = read_csv(
        'ESC_TROP_Vårdkontakt_InkluderadeIndexBesök_2017_2018.csv',
        usecols=index_colmap.keys()
    ).rename(columns=index_colmap)

    index.admission_date = pd.to_datetime(
        index.admission_date, format="%Y-%m-%d %H:%M:%S.%f")
    index = index.sort_values(by=['admission_date', 'ed_id'])
    index = index.drop_duplicates(subset=['Alias'], keep='first')
    index.male = index.male.apply(lambda x: 1 if x == 'M' else 0)

    index = index.dropna(subset=['cause']).drop(columns=['cause'])
    log.debug(f"{len(index)} unique chest-pain patients")

    # Load all lab-values from the index visit
    log.debug("Reading from lab-values")
    lab_colmap = {
        'KontaktId': 'ed_id',
        'Analyssvar_ProvtagningDatum': 'lab_date',
        'Labanalys_Namn': 'lab_name',
        'Analyssvar_Varde': 'value'
    }
    lab = read_csv(
        'ESC_TROP_LabAnalysSvar_InkluderadeIndexBesök_2017_2018.csv',
        usecols=lab_colmap.keys(),
    )
    lab = lab.rename(columns=lab_colmap)
    lab.lab_date = pd.to_datetime(
        lab.lab_date, format="%Y-%m-%d %H:%M:%S.%f")
    lab = lab.sort_values(by=['ed_id', 'lab_date'])

    # Map the KontaktId to Alias, and calculate time-difference
    lab = lab.merge(index, left_on='ed_id', right_on='ed_id', how='inner')
    lab['dt'] = (lab.lab_date - lab.admission_date).dt.total_seconds()
    lab = lab.loc[:, ['Alias', 'lab_name', 'value', 'dt']]

    # Extract the troponin, creatinine, hemoglobin and glucose lab-values
    # and merge them into a single dataframe.
    log.debug("Extracting troponin, creatinin, hemoglobin and glucose")
    final_lab_values = pd.concat(
        [
            _make_lab_value(lab, 'P-Troponin', new_name='troponin',
                            impute={'<5': 4.0}),
            _make_lab_value(lab, 'P-Kreatini', new_name='creatinine'),
            _make_lab_value(lab, 'B-Hemoglob', new_name='hemoglobin'),
            _make_lab_value(lab, 'P-Glukos', new_name='glucose')
        ],
        axis=1
    )

    features = (
        final_lab_values
        .dropna(how='any')
        .sort_index()
        .join(
            index
            .set_index('Alias')
            .loc[:, ['male', 'age', 'admission_date', 'hospital']]
        )
    )
    log.debug(f"{len(features)} patients with all lab-values")

    ecg_paths = _make_ecg_paths(features[['admission_date']])[['ecg_path']]
    features = features.join(ecg_paths).dropna()
    log.debug(f"{len(features)} patients with index ECG")

    log.debug("Calculating AMI/death label")
    mace_melior = make_mace_table(
        features[['admission_date']],
        include_interventions=False,
        include_deaths=True,
        use_melior=True,
        use_sos=False,
        use_hia=False,
    )

    features['ami30_melior'] = (
        mace_melior[["I21", "I22", "death"]]
        .any(axis=1)
        .astype(int)
    )

    mace_sos_hia = make_mace_table(
        features[['admission_date']],
        include_interventions=False,
        include_deaths=True,
        use_melior=False,
        use_sos=True,
        use_hia=True,
    )
    features['ami30_sos_hia'] = (
        mace_sos_hia[["I21", "I22", "death"]]
        .any(axis=1)
        .astype(int)
    )
    return features


def make_lindow_dataframe():
    """
    Create a dataframe with Alias as index and columns for TnT and index-ECG.
    The ECG columns are:
    * LBBB - Whether Glasgow thinks the ECG indicates LBBB
    * Incomplete_LBBB - Whether Glasgow thinks the ECG indicates partial LBBB
    * OverallQRSdur - Aggregated QRS durations (glasgow)
    * QRSdur_{lead} - QRS duration for all 12 leads individually
    * QRS_above_120 - True if any of the QRSdur_{lead} columns are true
    * path - Complete path to the original matlab-file

    The dataframe also contains, for convenience:
    * admission_date
    * age
    * sex
    * tnt_1 - (1st troponin)
    * tnt_date_1 - (timestamp for 1st troponin)
    * tnt_2
    * tnt_date_2
    * ecg - index of ECG in the hdf5-file
    * ecg_date
    """
    index_visits = read_csv(
        "ESC_TROP_Vårdkontakt_InkluderadeIndexBesök_2017_2018.csv",
        usecols=[
            'BesokOrsakId', 'Alias', 'Vardkontakt_InskrivningDatum',
            'KontaktId', 'Kön', 'Ålder vid inklusion'
        ],
        parse_dates=[
            'Vardkontakt_InskrivningDatum'
        ]
    )
    index_visits = (
        index_visits
        .dropna(subset=['BesokOrsakId'])
        .drop(columns=['BesokOrsakId'])
        .rename(columns={
            'Vardkontakt_InskrivningDatum': 'admission_date',
            'KontaktId': 'id',
            'Kön': 'sex',
            'Ålder vid inklusion': 'age', })
        .sort_values(by=['admission_date', 'id'])
        .drop_duplicates(subset=['Alias'], keep='first')
        .set_index('Alias')
        .join(make_troponin_table(), on='id', how='left')
        .drop(columns=['id', 'log_tnt_1', 'log_tnt_2'])
    )

    ecg_table = (
        make_ecg_table()
        .rename_axis('ecg')
        .reset_index()
        .set_index('Alias')
    )

    index_visits = index_visits.join(ecg_table, how='left').reset_index()
    index_visits['dt'] = (
        (index_visits.ecg_date - index_visits.admission_date)
        .dt.total_seconds().abs()
    )
    index_visits = (
        index_visits[index_visits.dt <= 3600]
        .sort_values(by=['Alias', 'dt'])
        .drop_duplicates(subset=['Alias'], keep='first')
    )
    index_visits.ecg = index_visits.ecg.astype(int)
    ecgs = index_visits.set_index('Alias').ecg.sort_values()

    qrs_overall = []
    diagnoses = []
    glasgow_strings = []

    def combine_diagnoses_to_one_string(ds):
        return ' --- '.join(
            [glasgow_diagnoses[i] for i, x in enumerate(ds) if x])

    with h5py.File(ECG_PATH, 'r') as ecg:
        OVERALL_QRS = glasgow_scalar_names.index('OverallQRSdur')
        ILBBB = glasgow_diagnoses.index('Incomplete LBBB')
        LBBB = glasgow_diagnoses.index('Left bundle branch block')
        for ecg_id in tqdm(ecgs):
            qrs_overall.append(ecg['glasgow']['scalars'][ecg_id, OVERALL_QRS])
            diagnoses.append(
                ecg['glasgow']['diagnoses'][ecg_id, [ILBBB, LBBB]]
            )
            glasgow_strings.append(
                combine_diagnoses_to_one_string(
                    ecg['glasgow']['diagnoses'][ecg_id, :]
                )
            )

    durations = _extract_glasgow_vector('QRSduration', ecgs)
    overall_duration = pd.DataFrame(
        qrs_overall,
        columns=['OverallQRSdur'],
        index=ecgs.index
    )
    durations['QRS_above_120'] = (overall_duration > 120).any(axis=1)

    lbbb = pd.DataFrame(
        np.stack(diagnoses),
        columns=['Incomplete_LBBB', 'LBBB'],
        index=ecgs.index
    )
    glasgow_strings = pd.DataFrame(
        glasgow_strings,
        columns=['Glasgow diagnoses (combined)'],
        index=ecgs.index
    )

    glasgow_vectors = [
        "ST_amp",
        "Qamplitude",
        "Ramplitude",
        "Rnotch",
        "Rprim_amp",
        "Rbis_dur",
        "Samplitude",
        "Sprim_amp",
    ]
    glasgow_vector_dfs = [
        _extract_glasgow_vector(name, ecgs) for name in glasgow_vectors
    ]

    index_visits = (
        index_visits
        .set_index('Alias')
        .join(
            [glasgow_strings, lbbb, overall_duration, durations] +
            glasgow_vector_dfs)
        .drop(columns=['dt'])
    )

    angio = _load_angio()
    df = _add_angio_to_lindow_df(index_visits, angio)

    # df = df.drop(columns=['ecg'])
    return df


def _add_angio_to_lindow_df(lindow, angio):
    angio = angio.join(lindow['ecg_date'])
    dt = (angio['angio_date'] - angio['ecg_date']).dt.total_seconds() / 3600
    angio['abs_dt'] = dt.abs()
    angio = (
        angio.loc[(dt >= -24) & (dt < 24*7), :]
        .sort_values(by='abs_dt')
        .reset_index()
        .drop_duplicates(subset=['Alias'], keep='first')
        .set_index('Alias')
        .drop(columns=['ecg_date', 'abs_dt'])
    )
    angio['had_recent_angio'] = True
    df = lindow.join(angio)
    df['had_recent_angio'] = df['had_recent_angio'].fillna(False)
    return df


def _load_angio():
    cols = [
        'Alias', 'INTERDAT', 'SEGMENTDIAGNOSTICS', 'FYND', 'FYND_OJUSTERAD',
        'STENOS', 'INDIKATION', 'TIDPCI', 'TIDCABG'
    ] + [f'SEGMENT{i}' for i in range(1, 21)]

    data = read_csv(
        'ESC_TROP_SWEDEHEART_DAT221_sc_angiopci_pop1.csv',
        usecols=cols,
        parse_dates=['INTERDAT']
    )
    data = data.rename(columns={'INTERDAT': 'angio_date'})
    return data.set_index('Alias')


def _extract_glasgow_vector(name, ecgs):
    name_index = glasgow_vector_names.index(name)
    result = []
    with h5py.File(ECG_PATH, 'r') as ecg:
        for ecg_id in tqdm(ecgs, desc=f"Extracting {name}"):
            result.append(ecg['glasgow']['vectors'][ecg_id, name_index, :])

    return pd.DataFrame(
        np.stack(result),
        columns=[f'{name}_{lead}' for lead in expected_lead_names],
        index=ecgs.index
    ).astype(int)


def _extract_glasgow_scalars(ecgs):
    result = []
    with h5py.File(ECG_PATH, 'r') as ecg:
        for ecg_id in tqdm(ecgs, desc='Extracting scalars'):
            result.append(ecg['glasgow']['scalars'][ecg_id, :])

        column_names = ecg['meta']['glasgow_scalar_names'][:].astype(str)
    return pd.DataFrame(
        result,
        index=ecgs.index,
        columns=column_names
    ).sort_index(axis=1).astype(int)


def _extract_glasgow_strings(ecgs):
    def combine_diagnoses_to_one_string(ds):
        return '\n'.join(
            [glasgow_diagnoses[i] for i, x in enumerate(ds) if x])

    def combine_rhythms_to_one_string(rs):
        return '\n'.join(
            [glasgow_rhythms[i] for i, x in enumerate(rs) if x])

    glasgow_strings = []
    with h5py.File(ECG_PATH, 'r') as ecg:
        for ecg_id in tqdm(ecgs):
            glasgow_strings.append([
                combine_diagnoses_to_one_string(
                    ecg['glasgow']['diagnoses'][ecg_id, :]
                ),
                combine_rhythms_to_one_string(
                    ecg['glasgow']['rhythms'][ecg_id, :]
                ),
            ])

    return pd.DataFrame(
        glasgow_strings,
        index=ecgs.index,
        columns=['glasgow_diagnoses', 'glasgow_rhythms']
    )


def _make_glasgow_pacemaker(ecgs):
    result = []
    with h5py.File(ECG_PATH, 'r') as ecg:
        rhythm_strings = list(
            ecg['meta']['glasgow_rhythms_names'][:].astype(str))
        diagnosis_strings = list(
            ecg['meta']['glasgow_diagnoses_names'][:].astype(str))
        pacemaker_rhythm_index = rhythm_strings.index(
            'A-V sequential pacemaker')
        pacemaker_diagnosis_index = diagnosis_strings.index(
            'Pacemaker rhythm - no further analysis')

        for ecg_id in tqdm(ecgs, desc='Making Glasgow pacemaker'):
            result.append([
                ecg['glasgow']['diagnoses'][ecg_id, pacemaker_diagnosis_index],
                ecg['glasgow']['rhythms'][ecg_id, pacemaker_rhythm_index]
            ])

    result = pd.DataFrame(
        result,
        index=ecgs.index,
        columns=['pacemaker_diagnosis', 'pacemaker_rhythm']
    )
    result['pacemaker'] = result.any(axis=1)
    return result


def _make_glasgow_lvh(ecgs):
    lvh_diagnoses = [
        'Ant/septal and lateral ST abnormality is probably due to the '
        'ventricular hypertrophy',
        'Ant/septal and lateral ST-T abnormality is probably due to the '
        'ventricular hypertrophy',
        'Ant/septal and lateral T wave abnormality is probably due to the '
        'ventricular hypertrophy',
        'Anterior T wave abnormality is probably due to the ventricular '
        'hypertrophy',
        'Anterolateral ST-T abnormality is probably due to the ventricular '
        'hypertrophy',
        'Anterolateral T wave abnormality is probably due to the ventricular '
        'hypertrophy',
        'Anteroseptal ST abnormality is probably due to the ventricular '
        'hypertrophy',
        'Anteroseptal ST-T abnormality is probably due to the ventricular '
        'hypertrophy',
        'Anteroseptal T wave abnormality is probably due to the ventricular '
        'hypertrophy',
        'Biventricular hypertrophy',
        'Inferior and ant/septal ST-T abnormality is probably due to the '
        'ventricular hypertrophy',
        'Inferior and ant/septal T wave abnormality is probably due to the '
        'ventricular hypertrophy',
        'Inferior and anterior ST-T abnormality is probably due to the '
        'ventricular hypertrophy',
        'Inferior and anterior T wave abnormality is probably due to the '
        'ventricular hypertrophy',
        'Inferior and septal T wave abnormality is probably due to the '
        'ventricular hypertrophy',
        'Inferior ST abnormality is probably due to the ventricular '
        'hypertrophy',
        'Inferior ST-T abnormality is probably due to the ventricular '
        'hypertrophy',
        'Inferior T wave abnormality is probably due to the ventricular '
        'hypertrophy',
        'Inferior/lateral ST abnormality is probably due to the '
        'ventricular hypertrophy',
        'Inferior/lateral T wave abnormality is probably due to the '
        'ventricular hypertrophy',
        'Lateral ST abnormality is probably due to the ventricular '
        'hypertrophy',
        'Lateral ST-T abnormality is probably due to the ventricular '
        'hypertrophy',
        'Lateral T wave abnormality is probably due to the ventricular '
        'hypertrophy',
        'Left ventricular hypertrophy',
        'Left ventricular hypertrophy by voltage only',
        'Possible biventricular hypertrophy',
        'Right ventricular hypertrophy',
        'Septal and lateral ST-T abnormality is probably due to the '
        'ventricular hypertrophy',
        'Septal ST abnormality is probably due to the ventricular hypertrophy',
        'Septal ST-T abnormality is probably due to the ventricular '
        'hypertrophy',
        'Septal T wave abnormality is probably due to the ventricular '
        'hypertrophy',
        'Widespread ST-T abnormality is probably due to the ventricular '
        'hypertrophy',
        'Widespread T wave abnormality is probably due to the ventricular '
        'hypertrophy',
    ]
    lvh = _make_glasgow_diagnosis_features(
        ecgs, lvh_diagnoses, desc='Making Glasgow LVH')
    lvh['lvh'] = lvh.any(axis=1)
    return lvh


def _make_glasgow_lbbb(ecgs):
    lbbb_diagnoses = [
        'Left bundle branch block'
    ]
    lbbb = _make_glasgow_diagnosis_features(
        ecgs, lbbb_diagnoses, desc='Making Glasgow LBBB')
    lbbb['lbbb'] = lbbb.any(axis=1)
    return lbbb


def _make_glasgow_rbbb(ecgs):
    rbbb_diagnoses = [
        'RBBB with left anterior fascicular block',
        'RBBB with RAD - possible left posterior fascicular block',
        'Right bundle branch block',
    ]
    rbbb = _make_glasgow_diagnosis_features(
        ecgs, rbbb_diagnoses, desc='Making Glasgow RBBB')
    rbbb['rbbb'] = rbbb.any(axis=1)
    return rbbb


def _make_glasgow_diagnosis_features(
        ecgs, diagnoses, desc='Extracting glasgow features'):
    result = []
    with h5py.File(ECG_PATH, 'r') as ecg:
        diagnosis_strings = list(
            ecg['meta']['glasgow_diagnoses_names'][:].astype(str))
        if any([diagnose not in diagnosis_strings for diagnose in diagnoses]):
            raise ValueError("Input diagnoses must match those in "
                             "glasgow_diagnoses_names")
        indices = [diagnosis_strings.index(diagnose) for diagnose in diagnoses]
        for ecg_id in tqdm(ecgs, desc=desc):
            result.append(ecg['glasgow']['diagnoses'][ecg_id, indices])

    return pd.DataFrame(
        result,
        index=ecgs.index,
        columns=diagnoses
    )


def _make_glasgow_rhythm_scores(ecgs):
    rhythm_score_lists = [
        [
            'Marked sinus bradycardia',
            'Sinus arrhythmia',
            'Sinus bradycardia',
            'Sinus rhythm',
            'Sinus tachycardia',
            'sinus arrhythmia',
            'Probable sinus tachycardia'
        ],
        [
            'Atrial fibrillation',
            'Atrial flutter',
            'Probable atrial fibrillation',
        ],
        [
            'A-V sequential pacemaker',
            'Atrial pacing',
            'Demand atrial pacing',
            'Demand pacing',
            'Ventricular pacing'
        ],
        [
            'Wide QRS tachycardia - possible ventricular tachycardia',
            'non-sustained ventricular tachycardia',
        ],
        [
            '2:1 A-V block',
            '2nd degree (Mobitz II) SA block',
            '2nd degree A-V block, Mobitz I (Wenckebach)',
            '2nd degree A-V block, Mobitz II',
            '3:1 A-V block',
            '4:1 A-V block',
            'complete A-V block',
            'high degree A-V block',
            'varying 2nd degree A-V block',
            'A-V dissociation'
        ],
        [
            '1st degree A-V block',
            'borderline 1st degree A-V block'
        ],
        [
            'Possible ectopic atrial tachycardia',
            'Probable atrial tachycardia',
            'Probable supraventricular tachycardia',
            'Regular supraventricular rhythm',
            'Wide QRS tachycardia - possible supraventricular tachycardia',
            'Irregular ectopic atrial tachycardia'
        ],
        [
            'Accelerated idioventricular rhythm',
            'Possible idioventricular rhythm',
            'paroxysmal idioventricular rhythm',
            'paroxysmal idioventricular rhythm or aberrant ventricular '
            'conduction',
        ],
        [
            'Possible accelerated junctional rhythm',
            'Possible junctional rhythm',
            'Probable accelerated junctional rhythm',
            'Probable junctional rhythm',
        ]
    ]

    all_rhythms = []
    with h5py.File(ECG_PATH, 'r') as ecg:
        glasgow_rhythm_names = list(
            ecg['meta']['glasgow_rhythms_names'][:].astype(str))
        rhythm_index = [
            [glasgow_rhythm_names.index(rhythm) for rhythm in rhythms]
            for rhythms in rhythm_score_lists
        ]
        for ecg_id in tqdm(ecgs, desc='Making glasgow rhythm scores'):
            all_rhythms.append(ecg['glasgow']['rhythms'][ecg_id, :])

    result = pd.DataFrame(all_rhythms, index=ecgs.index)
    for k, index in enumerate(rhythm_index):
        result[f"rhythm_score_{k+1}"] = result.iloc[:, index].any(axis=1)

    return result.iloc[:, len(glasgow_rhythm_names):]


def _make_ecg_paths(index):
    """
    Given a dataframe with Alias as index and admission_date as column,
    return a dataframe with Alias as index and the path to the index ECG for
    each corresponding patient.
    """
    log.debug("Processing ECGs")
    ecg = _make_double_ecg(
        index, min_age_seconds=-3600, max_age_seconds=4*3600
    )

    # ecg is now a dataframe with Alias as index and the hdf5 index of the
    # ECG as value in the column ecg_0. We want to map that index to
    # the path of the original ECG file.
    log.debug("Retrieving ECG paths")
    ecg = ecg[['ecg_0']].dropna().sort_values(by='ecg_0')
    with h5py.File(ECG_PATH, 'r') as ecg_hdf5:
        # Important that the index (ecg.values) is sorted here, because hdf5
        ecg['ecg_path'] = ecg_hdf5['meta']['path'][ecg.values]

    return ecg


def _make_lab_value(
        all_lab_values, lab_name, new_name=None, max_age_seconds=4*3600,
        impute=None):
    """
    Given a dataframe of all lab-values, extract a specific lab-value,
    optionally fill in some bad or missing values, convert the values to
    numeric format, and return only the first instance of a lab-value for
    each patient. If the age of a lab-value is above max_age_seconds, then
    it will not be used. Returns a series with Alias as index and the
    numeric lab-values as value. Only one value per patient.

    :param all_lab_values:
    :param lab_name:
    :param new_name:
    :param max_age_seconds:
    :param impute:
    :return:
    """
    matches_lab_name = all_lab_values.lab_name.str.fullmatch(lab_name)
    lab_values = all_lab_values.loc[matches_lab_name, :]

    if impute is not None:
        for nan, imputed_value in impute.items():
            missing_values = lab_values.value.str.match(nan)
            lab_values.loc[missing_values, 'value'] = imputed_value

    if new_name is None:
        new_name = lab_name

    lab_values.value = pd.to_numeric(lab_values.value, errors='coerce')
    lab_values.loc[lab_values.dt > max_age_seconds, 'value'] = np.nan
    lab_values = (
        lab_values
        .dropna()
        .sort_values(by=['Alias', 'dt'])
        .drop_duplicates(subset=['Alias'], keep='first')
        .set_index('Alias')
        .value
        .rename(new_name)
    )
    return lab_values


def _make_scaar_index(index, days_before=1, days_after=7):
    scaar = (
        read_csv(
            'ESC_TROP_SWEDEHEART_DAT221_sc_angiopci_pop1.csv',
            parse_dates=['INTERDAT'],
            usecols=['INTERDAT', 'SID_pseudo', 'MCEID_pseudo', 'Alias',
                     'EVENT'])
        .rename(columns={'INTERDAT': 'angio_date'})
        .dropna(how='all')
    )

    scaar_index = scaar.join(
        index[['admission_date', 'id']],
        how='inner',
        on='Alias'
    )
    dt = (scaar_index.angio_date - scaar_index.admission_date.dt.floor('D')
          ).dt.total_seconds() / (3600*24)
    scaar_index['distance'] = dt.abs()
    scaar_index = (
        scaar_index[(dt <= days_after) & (dt >= -days_before)]
        .sort_values(by=['Alias', 'distance'])
        .set_index('Alias')
        .drop(columns=['distance'])
    )
    return scaar_index


def _make_occlusion_and_presentation(index):
    log.debug('Gathering data on occlusion')
    sc_segment = read_csv(
        'ESC_TROP_SWEDEHEART_DAT221_sc_segment_pop1.csv',
        usecols=['SID_pseudo', 'OCKL', 'OCKLUSION']
    )
    sc_segment['occlusion_less_than_3_months_old'] = \
        sc_segment.OCKL == 'Ja, <3 mån'
    sc_segment['acute_presentation'] = (
        sc_segment.OCKLUSION == 'Ja, akut presentation (t.ex. SAT)')
    sc_segment['suspected_thrombosis'] = (
        sc_segment.OCKLUSION.isin([
            'Ja, akut presentation (t.ex. SAT)',
            'Nej, men misstänkt tromb'])
    )
    sc_segment['no_occlusion_suspected_thrombosis'] = (
        sc_segment.OCKLUSION == 'Nej, men misstänkt tromb')

    # Both OCKL and OCKLUSION columns sometimes have conflicting entries.
    # Here I only require that one of the entries satisfies the condition.
    sc_segment = sc_segment.groupby('SID_pseudo')[[
        'occlusion_less_than_3_months_old',
        'acute_presentation',
        'no_occlusion_suspected_thrombosis',
        'suspected_thrombosis'
    ]].any()

    return (
        index[['SID_pseudo']]
        .join(sc_segment, on='SID_pseudo')
        .fillna(False)
        .drop(columns=['SID_pseudo'])
        .groupby('Alias')
        .any()
    )


def _make_stenosis(index):
    log.debug('Gathering data on stenosis')
    stenosis_levels = [
        '0-29%', '30-49%', '50-69%', '70-89%', '90-99%', '100%'
    ]

    def collate_degrees_of_stenosis(s):
        return (
            pd.concat([(s == x).rename(x) for x in stenosis_levels], axis=1)
            .groupby(level='SID_pseudo')
            .any()
        )

    angiopci = (
        read_csv('ESC_TROP_SWEDEHEART_DAT221_sc_angiopci_pop1.csv')
        .dropna(how='all')
        .filter(regex='SEGMENT(\\d+)|(SID_pseudo)')
    )
    angiopci = pd.wide_to_long(
        angiopci, stubnames=['SEGMENT'], i='SID_pseudo', j='segment')

    finding = (
        read_csv('ESC_TROP_SWEDEHEART_DAT221_sc_finding_pop1.csv',
                 usecols=['SID_pseudo', 'STENOSGRAD'])
        .dropna(how='all')
        .set_index('SID_pseudo')
    )

    stenosis = (
        pd.concat(
            map(collate_degrees_of_stenosis,
                [angiopci.SEGMENT, finding.STENOSGRAD]),
            axis=1,
            keys=['angio', 'finding'],
            names=['source', 'stenosis'],
            join='outer')
        .fillna(False)
        .T.groupby(level='stenosis')  # groupby(axis=1) is deprecated
        .any().T
    )
    stenosis = (
        index[['SID_pseudo']]
        .join(stenosis, on='SID_pseudo')
        .drop(columns=['SID_pseudo'])
        .groupby('Alias')
        .any()
        .fillna(False)
    )
    max_stenosis = stenosis[reversed(stenosis_levels)].idxmax(axis=1)
    return (
        pd.get_dummies(max_stenosis)
        .loc[:, stenosis_levels]
        .rename(columns={x: f'stenosis_{x}' for x in stenosis_levels})
    )


def _make_acs_indication(index):
    log.debug('Gathering data on ACS indications')
    angiopci = (
        read_csv('ESC_TROP_SWEDEHEART_DAT221_sc_angiopci_pop1.csv',
                 usecols=['SID_pseudo', 'INDIKATION'])
        .set_index('SID_pseudo')
        .dropna(how='all')
    )
    acs_indication = (
        angiopci
        .INDIKATION.isin(['STEMI', 'NSTEMI', 'Instabil angina pectoris'])
        .rename('acs_indication')
    )
    return (
        index[['SID_pseudo']]
        .join(acs_indication, on='SID_pseudo')
        .drop(columns=['SID_pseudo'])
        .groupby('Alias')
        .any()
        .fillna(False)
    )


def _make_cabg(scaar_index):
    # Figure out if there was a CABG within 7 days of admission or
    # angiography. The scaar_index can have multiple angiographies for each
    # patient.
    log.debug('Gathering CABG data')
    angio = (
        read_csv(
            'ESC_TROP_SWEDEHEART_DAT221_sc_angiopci_pop1.csv',
            usecols=['Alias', 'SID_pseudo', 'INTERDAT', 'EVENT', 'CABG'],
            parse_dates=['INTERDAT'])
        .join(
            scaar_index[['angio_date', 'admission_date']],
            on='Alias',
            how='left')
        .sort_values(by=['Alias', 'INTERDAT'])
        .dropna(subset=['admission_date'])
    )
    dt = (angio.INTERDAT - angio.angio_date).dt.days
    cabg_angio = (
        angio[(dt >= 0) & (dt <= 7)]
        .groupby('Alias')
        .CABG.any()
        .rename('cabg_within_7_days_of_angio')
    )
    dt = (angio.INTERDAT - angio.admission_date.dt.floor('1D')).dt.days
    cabg_admission = (
        angio[(dt >= 0) & (dt <= 7)]
        .groupby('Alias')
        .CABG.any()
        .rename('cabg_within_7_days_of_admission')
    )

    return (
        pd.DataFrame(index=scaar_index.index)
        .join([cabg_angio, cabg_admission])
        .groupby('Alias')
        .any()
    )


def _make_rikshia_i21_tnt_cabg_stemi(scaar_index):
    log.debug('Gathering diagnosis and lab-values from rikshia')
    rikshia = read_csv(
        'ESC_TROP_SWEDEHEART_DAT221_rikshia_pop1.csv', dtype=str
    )
    rikshia = (
        scaar_index[['MCEID_pseudo']]
        .dropna()
        .join(rikshia.set_index('MCEID_pseudo'), on='MCEID_pseudo')
    )
    i21 = (
        rikshia
        .filter(regex='diag(\\d+)', axis=1)
        .stack()
        .str.match('I21')
        .rename('i21_rikshia')
        .reset_index(level=1, drop=True)
        .groupby('Alias')
        .any()
    )
    tnt = (
        rikshia.loc[
            rikshia.d_BIOCHEMICAL_MARKER_TYPE == 'HS Troponin-T (ng)',
            'd_BIOCHEMICAL_MARKER_VALUE']
        .astype(float)
        .dropna()
        .rename('tnt_rikshia')
    )
    cabg = (
        (rikshia.PRIOR_CARDIAC_SURGERY == 'CABG')
        .rename('prior_cabg')
    )
    stemi = (
        (rikshia.INFARCTTYPE == 'STEMI').rename('stemi_rikshia')
    )

    return (
        pd.DataFrame(index=scaar_index.index)
        .join([i21, tnt, cabg, stemi])
        .fillna({
            'prior_cabg': False,
            'i21_rikshia': False,
            'stemi_rikshia': False,
        })
        .groupby('Alias')
        .agg({
            'prior_cabg': 'any',
            'i21_rikshia': 'any',
            'stemi_rikshia': 'any',
            'tnt_rikshia': 'max'
        })
    )


def load_tnt():
    col_map = {
        'KontaktId': 'id',
        'Analyssvar_ProvtagningDatum': 'tnt_date',
        'Labanalys_Namn': 'name',
        'Analyssvar_Varde': 'tnt'
    }
    lab = read_csv(
        'ESC_TROP_LabAnalysSvar_InkluderadeIndexBesök_2017_2018.csv'
    )
    lab = lab.loc[:, list(col_map)].rename(columns=col_map)
    lab.tnt_date = pd.to_datetime(lab.tnt_date, format="%Y-%m-%d %H:%M:%S.%f")

    tnt = (
        lab.loc[
            lab.name.str.match('P-Troponin'),
            ['id', 'tnt', 'tnt_date']]
        .dropna()
        .sort_values(by=['id', 'tnt_date'])
    )

    tnt.loc[tnt.tnt.str.match('<5'), 'tnt'] = 4
    tnt.tnt = pd.to_numeric(tnt.tnt, errors='coerce')
    tnt = tnt.dropna()
    return tnt


def _make_melior_tnt(index):
    log.debug('Gathering TnT data from Melior')
    tnt = load_tnt()
    tnt_max = (
        tnt.sort_values(
            by=['id', 'tnt', 'tnt_date'],
            ascending=[True, False, True])
        .groupby('id').first()
    )
    return (
        index.join(tnt_max, on='id', how='left')[['tnt', 'tnt_date']]
        .rename(columns={'tnt': 'tnt_melior', 'tnt_date': 'tnt_melior_date'})
        .groupby('Alias')
        .max()
    )


def _make_sos_i21(index):
    log.debug('Gathering diagnoses from sos')
    sos = _make_diagnoses_from_sos().join(index)

    sos['i21_sos'] = sos.icd10.str.match('I21')

    dt = (sos.diagnosis_date - sos.admission_date.dt.floor('D')
          ).dt.total_seconds() / (3600 * 24)
    sos = (
        sos.loc[(dt >= -1) & (dt <= 1), 'i21_sos']
        .groupby('Alias')
        .any()
    )
    return sos.reindex(index.index).groupby('Alias').any()


def _make_sectra_angio(index):
    log.debug("Loading sectra files")
    sectra = load_sectra()

    log.debug("Finding angiographies")
    angiography_codes = ['37300', '39648', '39600']
    sectra_angio = (
        sectra
        .loc[sectra.sectra_code.isin(angiography_codes), :]
        .join(index.admission_date, how='inner', on='Alias')
    )
    dt = (sectra_angio.sectra_date -
          sectra_angio.admission_date).dt.total_seconds() / (3600 * 24)
    sectra_angio = (
        sectra_angio[(dt >= -1) & (dt <= 7)]
        .sort_values(by=['Alias', 'sectra_date'])
        .groupby('Alias')
        .agg({'sectra_date': 'first', 'answer': '\n\n'.join})
    )
    sectra_angio['has_angio'] = True
    return (
        pd.DataFrame(index=index.index)
        .join(sectra_angio)
        .fillna({'answer': 'N/A', 'has_angio': False})
    )


def make_omi_table(index):
    log.debug('Making index from scaar register')
    scaar_index = _make_scaar_index(index)
    pci_performed = (
        scaar_index
        .EVENT.isin(['PCI', 'PCIADHOC'])
        .rename('pci_performed')
        .groupby('Alias')
        .any()
    )
    omi = index.join(
        [
            _make_occlusion_and_presentation(scaar_index),
            _make_stenosis(scaar_index),
            _make_acs_indication(scaar_index),
            _make_rikshia_i21_tnt_cabg_stemi(scaar_index),
            _make_melior_tnt(index),
            _make_sos_i21(index),
            _make_cabg(scaar_index),
            pci_performed,
            _make_sectra_angio(index)
        ]
    ).fillna({
        'occlusion_less_than_3_months_old': False,
        'acute_presentation': False,
        'no_occlusion_suspected_thrombosis': False,
        'suspected_thrombosis': False,
        'stenosis_100%': False,
        'stenosis_90-99%': False,
        'stenosis_70-89%': False,
        'stenosis_50-69%': False,
        'stenosis_30-49%': False,
        'stenosis_0-29%': False,
        'acs_indication': False,
        'i21_rikshia': False,
        'prior_cabg': False,
        'stemi_rikshia': False,
        'i21_sos': False,
        'pci_performed': False,
        'cabg_within_7_days_of_angio': False,
        'cabg_within_7_days_of_admission': False
    })
    omi['tnt'] = omi[['tnt_melior', 'tnt_rikshia']].max(axis=1)
    omi['i21'] = omi.i21_rikshia | omi.i21_sos
    omi['stenosis_over_90%'] = omi['stenosis_100%'] | omi['stenosis_90-99%']
    omi['stenosis_under_70%'] = ~omi[[
        'stenosis_70-89%', 'stenosis_90-99%', 'stenosis_100%']].any(axis=1)
    omi['stenosis_70-99%'] = omi[[
        'stenosis_70-89%', 'stenosis_90-99%']].any(axis=1)

    return omi


def make_omi_label(omi_table, stenosis_limit=90, tnt_limit=750):
    stenosis_condition = (
        omi_table['stenosis_over_90%'] if stenosis_limit == 90 else
        omi_table['stenosis_100%']
    )

    omi = (
        omi_table.i21
        & (
            omi_table.occlusion_less_than_3_months_old
            | (omi_table.tnt >= tnt_limit)
            | (stenosis_condition & omi_table.acs_indication)
        )
    )
    return omi


def load_sectra():
    path = '/projects/air-crypt/air-crypt-raw/andersb/data/' \
           'ESC_Trop_17-18-2020-09-21/data/' \
           'ESC_TROP_sectra utdata_20190904.AB-Fnutt-fix.utf8.2023-05-09.csv'
    sectra = pd.read_csv(
        path,
        encoding='UTF-8',
        sep='|',
        quotechar='"',
        parse_dates=['Undersökningsdatum'],
        usecols=['Alias', 'Undersökningskod', 'Undersökningsdatum',
                 'Fullständigt_svar'],
        low_memory=False
    ).rename(columns={
        'Undersökningskod': 'sectra_code',
        'Undersökningsdatum': 'sectra_date',
        'Fullständigt_svar': 'answer'})
    return sectra
