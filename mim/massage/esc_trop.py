from os.path import join

import numpy as np
import pandas as pd
import h5py

from mim.massage.carlson_ecg import ECGStatus
from mim.util.logs import get_logger

log = get_logger("ESC-Trop Massage")


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
    ecg_path = '/mnt/air-crypt/air-crypt-esc-trop/axel/ecg.hdf5'
    with h5py.File(ecg_path, 'r') as ecg:
        table = pd.DataFrame(
            pd.to_datetime(
                ecg['meta']['date'][:],
                format="%Y-%m-%d %H:%M:%S"
            ),
            columns=['ecg_date']
        )
        table['Alias'] = '{' + pd.Series(
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


def _make_double_ecg(index):
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

    before = ecg.loc[(dt > -3600) & (dt < 0), :].drop_duplicates(
        subset=['Alias'], keep='last')
    after = ecg.loc[(dt >= 0) & (dt < 2*3600), :].drop_duplicates(
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


def make_ed_features(index):
    ed = _read_esc_trop_csv(
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
        .drop_duplicates(subset=['Alias'], keep='first')
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
    return ed
    # dt = (ed.ecg_date - ed.admission_date).dt.total_seconds()
    #
    # before = ed.loc[(dt > -3600) & (dt < 0), :].drop_duplicates(
    #     subset=['id'], keep='last')
    # after = ed.loc[(dt > 0) & (dt < 2*3600), :].drop_duplicates(
    #     subset=['id'], keep='first')
    #
    # table = (
    #     pd.concat([before, after], axis=0)
    #     .sort_values(by=['id', 'ecg_date'])
    #     .drop_duplicates(subset=['id'], keep='last')
    #     .rename_axis('Alias')
    #     .reset_index()
    #     .set_index('id')
    #  )
    #
    # old = (
    #     ed.loc[(dt < -7 * 24 * 3600), ['id', 'ecg_id', 'ecg_date']]
    #     .drop_duplicates(subset=['id'], keep='last')
    #     .rename(columns={'ecg_id': 'old_ecg_id', 'ecg_date': 'old_ecg_date'})
    #     .set_index('id')
    # )
    #
    # table = table.join(old, how='inner').sort_values(by='admission_date')
    # return table


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


def make_double_ecg_features(index):
    ecg = _make_double_ecg(index)
    ecg = index.join(ecg)
    ecg['delta_t'] = (ecg.admission_date - ecg.ecg_date_1).dt.total_seconds()
    ecg.delta_t /= 24 * 3600
    ecg['log_dt'] = (np.log10(ecg.delta_t) - 2.5) / 2  # Normalizing

    return ecg[['ecg_0', 'ecg_1', 'log_dt']]


def _read_esc_trop_csv(name):
    base_path = "/mnt/air-crypt/air-crypt-raw/andersb/data/" \
                "ESC_Trop_17-18-2020-09-21/data/"
    return pd.read_csv(
        join(base_path, name),
        encoding='latin1',
        sep='|'
    )


def make_mace_table(index_visits, include_interventions=True,
                    include_deaths=True, icd_definition='new',
                    use_melior=True, use_sos=True, use_hia=True):
    # Creates a table with index Alias, one row per index visit and one
    # column for each ICD/ATC code that is included in our definition of MACE
    # Also one column "mace30" which is true if any of the other columns are
    # true.
    # index_visits = make_index_visits()
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

    mace['mace30'] = mace.any(axis=1)
    return mace


def _make_diagnoses_from_sos():
    sos_sv = pd.read_csv(
        '/mnt/air-crypt/air-crypt-raw/andersb/data/'
        'Socialstyrelsen-2020-03-10/csv_files/sv_esctrop.csv'
    )
    sos_sv = sos_sv.dropna(subset=['Alias']).set_index('Alias')
    sos_sv['diagnosis_date'] = pd.to_datetime(sos_sv.INDATUM)

    diagnoses = (
        sos_sv.set_index('diagnosis_date', append=True)
        .filter(regex='(hdia)|(DIA[0-9]+)', axis=1)
        .stack()
        .rename('icd10')
        .reset_index(level=2, drop=True)
        .reset_index(level=1)
    )
    return diagnoses


def _make_diagnoses_from_rikshia():
    rikshia = _read_esc_trop_csv('ESC_TROP_SWEDEHEART_DAT221_rikshia_pop1.csv')

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


def _make_mace_diagnoses(index_visits, icds_defining_mace=None,
                         use_melior=True, use_sos=True, use_hia=True):
    if icds_defining_mace is None:
        icds_defining_mace = mace_codes_new

    assert any([use_melior, use_sos, use_hia])
    diagnose_sources = []

    if use_melior:
        diagnose_sources.append(_make_diagnoses(
            "ESC_TROP_Diagnoser_InkluderadeIndexBesök_2017_2018.csv"
        ))
        diagnose_sources.append(_make_diagnoses(
           "ESC_TROP_Diagnoser_EfterInkluderadeIndexBesök_2017_2018.csv"
        ))

    if use_sos:
        diagnose_sources.append(
            _make_diagnoses_from_sos()
        )

    if use_hia:
        diagnose_sources.append(
            _make_diagnoses_from_rikshia()
        )
    # diagnoses_deaths = _make_diagnoses_from_dors()

    # diagnoses is a DataFrame with index Alias and columns
    # "icd10" and "diagnosis_date". There is one row for each diagnosis, so
    # one patient can have multiple rows.
    diagnoses = pd.concat(diagnose_sources, axis=0).join(index_visits)

    # Calculate the time between admission and diagnosis, in days
    dt = (diagnoses.diagnosis_date.dt.floor('1D')) - \
         (diagnoses.admission_date.dt.floor('1D'))
    dt = dt.dt.total_seconds() / (3600 * 24)

    # We keep only the diagnoses within 30 days of the index visit
    # Maybe set lower span to something like -0.5?
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


def make_index_visits(exclude_stemi=True, exclude_missing_tnt=True,
                      exclude_missing_ecg=True, exclude_missing_old_ecg=True):
    """
    Creates a DataFrame with Alias as index and columns, admission_date and
    id, which corresponds to KontaktId in the original CSV-file.
    """
    index_visits = _read_esc_trop_csv(
        "ESC_TROP_Vårdkontakt_InkluderadeIndexBesök_2017_2018.csv"
    )
    index_visits = index_visits[[
        'Alias', 'Vardkontakt_InskrivningDatum', 'KontaktId']]
    index_visits = index_visits.rename(
        columns={
            'Vardkontakt_InskrivningDatum': 'admission_date',
            'KontaktId': 'id'
        }
    )
    n = len(index_visits)
    log.debug(f'{n} index visits loaded from file.')

    index_visits.admission_date = pd.to_datetime(index_visits.admission_date)
    index_visits = (
        index_visits
        .sort_values(by=['admission_date', 'id'])
        .drop_duplicates(subset=['Alias'], keep='first')
        .set_index('Alias')
    )
    n = len(index_visits)
    log.debug(f'{n} unique patients.')

    if exclude_stemi:
        stemi = index_visits.join(_make_index_stemi())
        dt = (stemi.admission_date - stemi.stemi_date
              ).dt.total_seconds() / (24*3600)
        index_stemi = stemi[(dt >= -1) & (dt <= 1)].index.unique()
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

    return index_visits.sort_values(by=['admission_date', 'id'])


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

    mace_kva_new = [
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
    ]
    for action in mace_kva_new:
        actions[action] = actions.action.str.contains(action)

    mace_actions = pd.DataFrame(index=index_visits.index)
    mace_actions = mace_actions.join(
        actions.groupby('Alias')[mace_kva_new].any(),
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


def _make_index_stemi():
    hia = _read_esc_trop_csv('ESC_TROP_SWEDEHEART_DAT221_rikshia_pop1.csv')
    hia = hia[
        ['Alias', 'ECG_STT_CHANGES', 'INFARCTTYPE', 'ADMISSION_ER_DATE',
         'ADMISSION_ER_TIME', 'ADMISSION_DATE', 'ADMISSION_TIME']]

    hia.ADMISSION_ER_DATE = hia.ADMISSION_ER_DATE.fillna(hia.ADMISSION_DATE)
    hia.ADMISSION_ER_TIME = hia.ADMISSION_ER_TIME.fillna(hia.ADMISSION_TIME)

    hia['stemi_date'] = pd.to_datetime(
        hia.ADMISSION_ER_DATE + ' ' + hia.ADMISSION_ER_TIME)
    hia = hia.set_index('Alias')

    return hia.loc[hia.INFARCTTYPE == 'STEMI', 'stemi_date']
