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
        'Alias': 'Alias',
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
        .drop_duplicates(subset=['Alias'], keep='last')
        .set_index('Alias')
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
        .rename_axis('Alias')
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

    # Exclude patients with STEMI according to HIA at index
    # This drops total from 19444 to 19251
    stemi = _make_index_stemi(_make_index_visits())
    ed = ed.set_index('Alias')[~stemi.index_stemi].reset_index()

    ed['delta_t'] = (ed.ecg_date - ed.old_ecg_date).dt.total_seconds()
    ed.delta_t /= 24 * 3600  # delta_t unit is now days
    ed.delta_t = (np.log10(ed.delta_t) - 2.5) / 2  # Normalizing
    ed.sex = ed.sex.apply(lambda x: 1 if x == 'M' else 0)

    return ed


def _read_esc_trop_csv(name):
    base_path = "/mnt/air-crypt/air-crypt-raw/andersb/data/" \
                "ESC_Trop_17-18-2020-09-21/data/"
    return pd.read_csv(
        join(base_path, name),
        encoding='latin1',
        sep='|'
    )


def make_mace_table(use_paul_icds=False, include_interventions=True,
                    include_deaths=True):
    # Creates a table with index Alias, one row per index visit and one
    # column for each ICD/ATC code that is included in our definition of MACE
    # Also one column "mace30" which is true if any of the other columns are
    # true.
    index_visits = _make_index_visits()
    sources_of_mace = []
    if use_paul_icds:
        sources_of_mace.append(
            _make_mace_diagnoses(
                index_visits,
                icds_defining_mace=mace_codes_paul)
        )
    else:
        sources_of_mace.append(_make_mace_diagnoses(index_visits))

    if include_interventions:
        sources_of_mace.append(_make_mace_interventions(index_visits))
    if include_deaths:
        sources_of_mace.append(_make_mace_deaths(index_visits))
    if len(sources_of_mace) > 1:
        mace = pd.concat(sources_of_mace, axis=1)
    else:
        mace = sources_of_mace[0]

    mace['mace30'] = mace.any(axis=1)
    return mace


def _make_mace_diagnoses(index_visits, icds_defining_mace=None):
    if icds_defining_mace is None:
        icds_defining_mace = mace_codes_new

    diagnoses_current = _make_diagnoses(
        "ESC_TROP_Diagnoser_InkluderadeIndexBesök_2017_2018.csv"
    )
    diagnoses_after = _make_diagnoses(
        "ESC_TROP_Diagnoser_EfterInkluderadeIndexBesök_2017_2018.csv"
    )
    diagnoses_deaths = _make_diagnoses_from_dors()

    # diagnoses is a DataFrame with index Alias and columns
    # "icd10" and "diagnosis_date". There is one row for each diagnosis, so
    # one patient can have multiple rows.
    diagnoses = pd.concat([
        diagnoses_current,
        diagnoses_after,
        diagnoses_deaths
    ], axis=0)

    diagnoses = diagnoses.join(index_visits)

    # Calculate the time between admission and diagnosis, in days
    dt = diagnoses.diagnosis_date - diagnoses.admission_date
    dt = dt.dt.total_seconds() / (3600 * 24)

    # We keep only the diagnoses within 30 days of the index visit
    diagnoses = diagnoses[(dt >= 0) & (dt <= 30)]

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


def _make_index_visits():
    """
    Creates a DataFrame with Alias as index and one column, admission_date.
    The admission_date is in datetime format and there is one row per visit,
    no duplicates.
    """
    index_visits = _read_esc_trop_csv(
        "ESC_TROP_Vårdkontakt_InkluderadeIndexBesök_2017_2018.csv"
    )
    index_visits = index_visits[['Alias', 'Vardkontakt_InskrivningDatum']]
    index_visits = index_visits.rename(
        columns={'Vardkontakt_InskrivningDatum': 'admission_date'}
    )

    index_visits.admission_date = pd.to_datetime(index_visits.admission_date)
    index_visits = (
        index_visits
        .sort_values(by='admission_date')
        .drop_duplicates(subset=['Alias'], keep='first')
        .set_index('Alias')
    )
    return index_visits


def _make_mace_interventions(index_visits):
    # combine tables
    # calculate dt
    # filter on dt
    # calculate new columns
    # groupby alias and combine columns
    # return results as new table
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
        "DF017",
        "DF025",
        "DF028",
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
    actions = _read_esc_trop_csv(csv_name)

    actions = actions[
        ["Alias", "PatientAtgard_Kod", "PatientAtgard_ModifieradDatum"]
    ]
    actions = actions.rename(
        columns={
            'PatientAtgard_Kod': 'action',
            'PatientAtgard_ModifieradDatum': 'date'
        }
    )
    actions['date'] = pd.to_datetime(actions['date'])
    return actions.set_index('Alias')


def _make_diagnoses(csv_name):
    diagnoses = _read_esc_trop_csv(csv_name)
    diagnoses = diagnoses[
        ['Alias', 'PatientDiagnos_Kod', 'PatientDiagnos_ModifieradDatum']
    ]
    diagnoses = diagnoses.rename(
        columns={
            'PatientDiagnos_Kod': 'icd10',
            'PatientDiagnos_ModifieradDatum': 'diagnosis_date'
        }
    )
    diagnoses.diagnosis_date = pd.to_datetime(diagnoses.diagnosis_date)
    return diagnoses.set_index('Alias')


def _make_mace_deaths(index_visits):
    deaths = _read_esc_trop_csv('ESC_TROP_SOS_R_DORS__14204_2019.csv')
    deaths = deaths.set_index('Alias')[['DODSDAT']].rename(
        columns={'DODSDAT': 'diagnosis_date'})
    deaths.diagnosis_date = pd.to_datetime(deaths.diagnosis_date,
                                           format='%Y%m%d', errors='coerce')
    deaths = index_visits.join(deaths)
    dt = (deaths.diagnosis_date - deaths.admission_date).dt.total_seconds() / (
                3600 * 24)
    deaths['death'] = (dt >= 0) & (dt <= 30)
    return deaths[['death']]


def _make_diagnoses_from_dors():
    deaths = _read_esc_trop_csv('ESC_TROP_SOS_R_DORS__14204_2019.csv')
    diagnoses = (
        deaths
        .set_index('Alias')
        .filter(regex='(ULORSAK)|(MORSAK)', axis=1)
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
        diagnoses.diagnosis_date,
        format='%Y%m%d',
        errors='coerce'
    )
    return diagnoses.dropna()


def _make_index_stemi(index_visits):
    hia = _read_esc_trop_csv('ESC_TROP_SWEDEHEART_DAT221_rikshia_pop1.csv')
    hia = hia[
        ['Alias', 'ECG_STT_CHANGES', 'ADMISSION_ER_DATE', 'ADMISSION_ER_TIME',
         'ADMISSION_DATE', 'ADMISSION_TIME']]

    hia.ADMISSION_ER_DATE = hia.ADMISSION_ER_DATE.fillna(hia.ADMISSION_DATE)
    hia.ADMISSION_ER_TIME = hia.ADMISSION_ER_TIME.fillna(hia.ADMISSION_TIME)

    hia['hia_date'] = pd.to_datetime(
        hia.ADMISSION_ER_DATE + ' ' + hia.ADMISSION_ER_TIME)
    hia = hia.set_index('Alias')
    hia = index_visits.join(hia)
    dt_hours = (hia.hia_date - hia.admission_date).dt.total_seconds() / 3600

    hia['index_stemi'] = (dt_hours >= -24) & (dt_hours <= 24) & (
                hia.ECG_STT_CHANGES == 'ST-höjning')

    index_stemi = (
        hia.reset_index()[['Alias', 'index_stemi']]
        .sort_values(by=['Alias', 'index_stemi'])
        .drop_duplicates(subset='Alias', keep='last')
        .set_index('Alias')
    )
    return index_stemi
