from os.path import join

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

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


def make_forberg_features(ecg_ids):
    ecg_path = '/mnt/air-crypt/air-crypt-esc-trop/axel/ecg.hdf5'
    with h5py.File(ecg_path, 'r') as ecg:
        glasgow_features = []
        for x in tqdm(ecg_ids, desc='Extracting Glasgow vectors'):
            glasgow_features.append(ecg['glasgow']['vectors'][x])

        vector_names = list(ecg['meta']['glasgow_vector_names'][:])
        lead_names = ecg['meta']['lead_names'][:]

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


def make_troponin_table():
    col_map = {
        'KontaktId': 'ed_id',
        'Analyssvar_ProvtagningDatum': 'tnt_date',
        'Labanalys_Namn': 'name',
        'Analyssvar_Varde': 'tnt'
    }
    lab = _read_esc_trop_csv(
        'ESC_TROP_LabAnalysSvar_InkluderadeIndexBesök_2017_2018.csv'
    )
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
    tnt.tnt = pd.to_numeric(tnt.tnt, errors='coerce')
    tnt = tnt.dropna()

    first_tnt = tnt.drop_duplicates(subset=['ed_id'], keep='first')
    second_tnt = tnt[~tnt.index.isin(first_tnt.index)].drop_duplicates(
        subset=['ed_id'], keep='first')

    r = first_tnt.set_index('ed_id').join(
            second_tnt.set_index('ed_id'),
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


def _read_esc_trop_csv(name, **kwargs):
    base_path = "/mnt/air-crypt/air-crypt-raw/andersb/data/" \
                "ESC_Trop_17-18-2020-09-21/data/"
    return pd.read_csv(
        join(base_path, name),
        encoding='latin1',
        sep='|',
        **kwargs
    )


def make_pauls_mace(index_visits):
    mace = _read_esc_trop_csv(
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
        'DF', 'DG', 'FN', 'FP', 'TF', 'death'
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
        '/mnt/air-crypt/air-crypt-raw/andersb/data/'
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
    rikshia = _read_esc_trop_csv(
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
        exclude_missing_chest_pain=True):
    """
    Creates a DataFrame with Alias as index and columns, admission_date and
    id, which corresponds to KontaktId in the original CSV-file.
    """
    index_visits = _read_esc_trop_csv(
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
    for action in mace_kva:
        actions[action] = actions.action.str.contains(action)

    mace_actions = pd.DataFrame(index=index_visits.index)
    mace_actions = mace_actions.join(
        actions.groupby('Alias')[mace_kva].any(),
        how='left'
    )
    return mace_actions.fillna(False)


def _make_actions(csv_name):
    actions = _read_esc_trop_csv(
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
    diagnoses = _read_esc_trop_csv(
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


def _fix_dors_date(s):
    if s[-4:] == '0000':
        return s[:-4] + '1201'
    elif s[-2:] == '00':
        return s[:-2] + '01'
    else:
        return s


def _make_mace_deaths(index_visits):
    deaths = _read_esc_trop_csv('ESC_TROP_SOS_R_DORS__14204_2019.csv')
    deaths = deaths.set_index('Alias')[['DODSDAT']].rename(
        columns={'DODSDAT': 'diagnosis_date'})

    deaths.diagnosis_date = pd.to_datetime(
        deaths.diagnosis_date.astype(str).apply(_fix_dors_date),
        format='%Y%m%d'
    )
    deaths = index_visits.join(deaths)
    dt = (deaths.diagnosis_date -
          deaths.admission_date).dt.total_seconds() / (3600 * 24)
    deaths['death'] = (dt >= -30) & (dt <= 30)
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
        diagnoses.diagnosis_date.astype(str).apply(_fix_dors_date),
        format='%Y%m%d',
    )
    return diagnoses.dropna()


def _make_index_stemi():
    hia = _read_esc_trop_csv(
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

    index = _read_esc_trop_csv(
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
    lab = _read_esc_trop_csv(
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
    hdf5_path = '/mnt/air-crypt/air-crypt-esc-trop/axel/ecg.hdf5'
    with h5py.File(hdf5_path, 'r') as ecg_hdf5:
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
