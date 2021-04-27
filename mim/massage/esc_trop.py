from os.path import join

import numpy as np
import pandas as pd
import h5py

from mim.massage.carlson_ecg import ECGStatus


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
    ECGStatus.MISSING_PATIENT
}


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


mace_codes_paul2 = [
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
    ecg_path = '/mnt/air-crypt/air-crypt-esc-trop/axel/ecg.hdf5'
    with h5py.File(ecg_path, 'r') as ecg:
        table = pd.DataFrame(
            pd.to_datetime(
                ecg['meta']['date'][:],
                format="%Y-%m-%d %H:%M:%S"
            ),
            columns=['ecg_date']
        )
        table['patient_id'] = '{' + pd.Series(
            ecg['meta']['alias'][:]
        ).str.upper() + '}'

        status = pd.DataFrame(
            ecg['meta']['status'][:],
            columns=ecg['meta']['status_keys'][:]
        )
        status['USABLE'] = ~status[list(map(
            lambda s: s.name, important_status_labels))].any(axis=1)

        table = table.loc[status.USABLE & (table.ecg_date >= '1999')]
        return table


def make_ed_table():
    ed_path = '/mnt/air-crypt/air-crypt-raw/andersb/data/ESC_Trop_' \
              '17-18-2020-09-21/data/ESC_TROP_Vårdkontakt_' \
              'InkluderadeIndexBesök_2017_2018.csv'

    col_map = {
        'KontaktId': 'id',
        'Alias': 'alias',
        'Vardkontakt_InskrivningDatum': 'admission_date',
        'Kön': 'sex',
        'Ålder vid inklusion': 'age',
        'MACE inom 30 dagar': 'mace_30_days'
    }

    ed = pd.read_csv(ed_path, encoding='latin1', sep='|')
    ed = ed.loc[:, list(col_map)].rename(columns=col_map)
    ed.admission_date = pd.to_datetime(
        ed.admission_date, format="%Y-%m-%d %H:%M:%S.%f")

    ed = (
        ed.sort_values(by='admission_date')
        .drop_duplicates(subset=['alias'], keep='last')
        .set_index('alias')
    )
    ecg = (
        make_ecg_table()
        .rename_axis('ecg_id')
        .reset_index()
        .set_index('patient_id')
    )

    ed = ed.join(ecg, how='left')
    ed = ed.dropna(subset=['ecg_date']).sort_values(by=['id', 'ecg_date'])
    dt = (ed.ecg_date - ed.admission_date).dt.total_seconds()

    before = ed.loc[(dt > -3600) & (dt < 0), :].drop_duplicates(
        subset=['id'], keep='last')
    after = ed.loc[(dt > 0) & (dt < 2*3600), :].drop_duplicates(
        subset=['id'], keep='first')

    table = (
        pd.concat([before, after], axis=0)
        .sort_values(by=['id', 'ecg_date'])
        .drop_duplicates(subset=['id'], keep='last')
        .rename_axis('alias')
        .reset_index()
        .set_index('id')
     )

    old = (
        ed.loc[(dt < -7 * 24 * 3600), ['id', 'ecg_id', 'ecg_date']]
        .drop_duplicates(subset=['id'], keep='last')
        .rename(columns={'ecg_id': 'old_ecg_id', 'ecg_date': 'old_ecg_date'})
        .set_index('id')
    )

    table = table.join(old, how='inner').sort_values(by='admission_date')
    return table


def make_troponin_table():
    path = (
        '/mnt/air-crypt/air-crypt-raw/andersb/data/'
        'ESC_Trop_17-18-2020-09-21/data/'
        'ESC_TROP_LabAnalysSvar_InkluderadeIndexBesök_2017_2018.csv'
    )

    col_map = {
        'KontaktId': 'ed_id',
        'Analyssvar_ProvtagningDatum': 'tnt_date',
        'Labanalys_Namn': 'name',
        'Analyssvar_Varde': 'tnt'
    }

    lab = pd.read_csv(path, encoding='latin1', sep='|')
    lab = lab.loc[:, list(col_map)].rename(columns=col_map)
    lab.tnt_date = pd.to_datetime(lab.tnt_date, format="%Y-%m-%d %H:%M:%S.%f")

    tnt = (
        lab.loc[
            lab.name.str.match('P-Troponin'),
            ['ed_id', 'tnt', 'tnt_date']]
        .dropna()
        .sort_values(by=['ed_id', 'tnt_date'])
    )

    tnt.loc[tnt.tnt.str.match('<5'), 'tnt'] = 4
    tnt.tnt = pd.to_numeric(tnt.tnt, errors='coerce').dropna()

    first_tnt = tnt.drop_duplicates(subset=['ed_id'], keep='first')
    second_tnt = tnt[~tnt.index.isin(first_tnt.index)].drop_duplicates(
        subset=['ed_id'], keep='first')

    return (
        first_tnt.set_index('ed_id').join(
            second_tnt.set_index('ed_id'),
            lsuffix='_1',
            rsuffix='_2'
        )
    )


def make_double_ecg_table():
    ed = make_ed_table()
    tnt = make_troponin_table()
    ed = ed.join(tnt).reset_index()

    # Include only those that have a first valid tnt measurement!
    # This drops total from 20506 to 19444. There are 8722 patients with
    # two valid tnts.
    ed = ed.dropna(subset=['tnt_1'])

    ed['delta_t'] = (ed.ecg_date - ed.old_ecg_date).dt.total_seconds()
    ed.delta_t /= 24 * 3600  # delta_t unit is now days
    ed.delta_t = (np.log10(ed.delta_t) - 2.5) / 2  # Normalizing
    ed.sex = ed.sex.apply(lambda x: 1 if x == 'M' else 0)
    return ed


def make_mace_table(include_dors=True):
    def read_esc_trop_csv(name):
        base_path = "/mnt/air-crypt/air-crypt-raw/andersb/data/" \
                    "ESC_Trop_17-18-2020-09-21/data/"
        return pd.read_csv(
            join(base_path, name),
            encoding='latin1',
            sep='|'
        )

    index_visits = read_esc_trop_csv(
        "ESC_TROP_Vårdkontakt_InkluderadeIndexBesök_2017_2018.csv"
    )
    diagnoses_after = read_esc_trop_csv(
        "ESC_TROP_Diagnoser_EfterInkluderadeIndexBesök_2017_2018.csv"
    )
    diagnoses_current = read_esc_trop_csv(
        "ESC_TROP_Diagnoser_InkluderadeIndexBesök_2017_2018.csv"
    )
    deaths = read_esc_trop_csv('ESC_TROP_SOS_R_DORS__14204_2019.csv')

    mace = index_visits[
        ['Alias', 'Vardkontakt_InskrivningDatum', 'MACE inom 30 dagar']
    ].rename(
        columns={
            'Vardkontakt_InskrivningDatum': 'admission_date',
            'MACE inom 30 dagar': 'mace_paul'
        }
    )
    mace.admission_date = pd.to_datetime(mace.admission_date)
    mace = mace.sort_values(by='admission_date').drop_duplicates(
        subset=['Alias'], keep='first').set_index('Alias')

    mace_current = diagnoses_current[
        ['Alias', 'PatientDiagnos_Kod', 'PatientDiagnos_ModifieradDatum']
    ].rename(
        columns={
            'PatientDiagnos_Kod': 'icd10',
            'PatientDiagnos_ModifieradDatum': 'diagnosis_date'
        }
    )
    mace_current.diagnosis_date = pd.to_datetime(mace_current.diagnosis_date)
    mace_current = mace_current.set_index('Alias')

    mace_after = diagnoses_after[
        ['Alias', 'PatientDiagnos_Kod', 'PatientDiagnos_ModifieradDatum']
    ].rename(
        columns={
            'PatientDiagnos_Kod': 'icd10',
            'PatientDiagnos_ModifieradDatum': 'diagnosis_date'
        }
    )
    mace_after.diagnosis_date = pd.to_datetime(mace_after.diagnosis_date)
    mace_after = mace_after.set_index('Alias')

    deaths_diagnoses = deaths.set_index('Alias').filter(
        regex='(ULORSAK)|(MORSAK)', axis=1).stack().reset_index(level=1,
                                                                drop=True)
    deaths_diagnoses = pd.DataFrame(
        deaths_diagnoses.rename('icd10')
    ).join(deaths.set_index('Alias')[['DODSDAT']], how='left')
    deaths_diagnoses = deaths_diagnoses.rename(
        columns={'DODSDAT': 'diagnosis_date'})
    deaths_diagnoses['diagnosis_date'] = pd.to_datetime(
        deaths_diagnoses.diagnosis_date, format='%Y%m%d', errors='coerce')
    deaths_diagnoses = deaths_diagnoses.dropna()

    diagnoses_to_combine = [mace_current, mace_after]
    if include_dors:
        diagnoses_to_combine.append(deaths_diagnoses)
    mace_combined = pd.concat(diagnoses_to_combine, axis=0)
    mace_combined = mace_combined.join(mace[['admission_date']])
    mace_combined['dt'] = (mace_combined.diagnosis_date -
                           mace_combined.admission_date
                           ).dt.total_seconds() / (3600 * 24)

    mace30 = mace_combined[
        (mace_combined['dt'] >= 0) &
        (mace_combined['dt'] <= 30)
    ]
    mace_codes = mace_codes_paul
    for icd in mace_codes:
        mace30[icd] = mace30.icd10.str.contains(icd)

    mace = mace.join(mace30.groupby('Alias')[mace_codes].any(),
                     how='left').fillna(False)
    mace['mace30'] = mace.iloc[:, 2:].any(axis=1)
    return mace
