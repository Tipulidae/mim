from os.path import join

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from massage.sos_util import fix_dors_date
from mim.util.logs import get_logger
from mim.cache.decorator import cache
from massage.icd_util import round_icd_to_chapter, icd_chapters, \
    atc_to_level_rounder

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
        'Sjukhus_Namn': 'hospital',
        'outcome-30d-I200-SV': 'I200',
        'outcome-30d-I21-SV': 'I21',
        'outcome-30d-I22-SV': 'I22',
        'outcome-30d-DEATH': 'death',
    }
    data = (
        pd.read_csv(
            '/mnt/air-crypt/air-crypt-esc-trop/andersb/scratch/brsm-U.csv',
            parse_dates=['Vardkontakt_InskrivningDatum'],
            usecols=colmap.keys())
        .rename(columns=colmap)
        .sort_values(by=['Alias', 'admission_date'])
    )

    data['admission_index'] = range(len(data))
    data.admission_date = data.admission_date.dt.floor('D')

    return data


@cache
def multihot_icd_kva(source, index):
    log.info(
        f'Constructing multihot encoded ICD and KVÅ events from {source}')

    source = source.upper()
    assert source in ['SV', 'OV']

    path = f'SOS_T_T_T_R_PAR_{source}_24129_2020.csv'
    index = index.loc[:, ['Alias', 'LopNr']].set_index('LopNr')
    log.info(f'Loading {path}')
    diagnosis_cols = ['hdia'] + [f'DIA{x}' for x in range(1, 31)]
    op_cols = ['OP']
    sos = (
        read_csv(
            path,
            usecols=['LopNr', 'INDATUM'] + diagnosis_cols + op_cols,
            parse_dates=['INDATUM'],
            low_memory=False)
        .rename(columns={'INDATUM': 'event_date'})
        .join(index, on='LopNr', how='inner')
        .set_index(['Alias', 'event_date'])
        .sort_index()
    )

    icd = (
        sos
        .loc[:, diagnosis_cols]
        .stack()
        .rename('ICD')
        .reset_index(level=2, drop=True)
    )
    icd = pivot(icd)

    # icd.columns = [f'{source}_ICD_{col}' for col in icd.columns]

    op = (
        sos
        .loc[:, 'OP']
        .str.split(' ')
        .explode()
        .rename('OP')
        .dropna()
    )

    op = pivot(op)
    op.columns = [f'{source}_OP_{col}' for col in op.columns]
    return icd, op


@cache
def multihot_atc(index):
    log.info('Constructing multihot encoded ATC events')
    atc = (
        read_csv(
            'SOS_T_R_LMED_24129_2020.csv',
            usecols=['LopNr', 'ATC', 'EDATUM'],
            parse_dates=['EDATUM'])
        .rename(columns={'EDATUM': 'event_date'})
        .join(index.set_index('LopNr').Alias, on='LopNr', how='inner')
        .set_index(['Alias', 'event_date'])
        .sort_index()
        .loc[:, 'ATC']
    )
    atc = pivot(atc)
    # atc.columns = [f"ATC_{col}" for col in atc.columns]
    return atc


def pivot(s):
    # Input is a series with MultiIndex (Alias, event_date)
    # Values are ICD, ATC or KVÅ codes.
    # Output is a dataframe with the same index, but the values are one-hot
    # encoded columns, with one column for each unique value in the input
    # series. The columns are sorted by frequency, so that the most common
    # value in the input is the first column in the output.
    name = s.name
    log.info(f'Pivoting {name}')
    df = s.reset_index().drop_duplicates()
    col_order = (
        df[name]
        .value_counts()
        .reset_index()
        .sort_values(by=[name, 'index'], ascending=[False, True])
        .loc[:, 'index']
        .values
    )

    df['temp'] = True
    df = df.pivot_table(
        index=['Alias', 'event_date'],
        columns=name,
        values='temp'
    )
    return df.fillna(0).astype(bool).loc[:, col_order]


def stagger_events(df, brsm):
    """

    :param df: Dataframe with index (Alias, event_date)-tuple.
    :param brsm: Dataframe containing one row for each chest-pain visit.
    Required columns are Alias, admission_date and admission_index.
    :return: New version of the dataframe, but staggered such that each
    row corresponds to a single event-index pair. Thus, if a patient is
    represented twice in the index (brsm), each event preceding the index
    date will be included twice: once for each index-visit. Events
    occurring after the index-visits are removed.
    """
    log.info("Staggering events")
    brsm = brsm.set_index('Alias')[['admission_index', 'admission_date']]
    df = df.reset_index().join(brsm, on='Alias')
    years_ago = (
        (df.admission_date - df.event_date)
        .dt.total_seconds() / (3600 * 24 * 365)
    )
    df = (
        # df[df.event_date < df.admission_date]
        df[(years_ago <= 5) & (years_ago > 0)]
        .sort_values(
            by=['Alias', 'admission_index', 'event_date'])
        .reset_index(drop=True)
    )
    return df


def sum_events_in_interval(mhs, brsm, **interval_kwargs):
    """
    :param mh: Multihot-encoded matrix, with index (Alias, event_date).
    Each column correspond to some binary "event" (ICD, KVÅ or ATC code).
    :param brsm: Dataframe containing one row for each chest-pain visit.
    Required columns are Alias, admission_date and admission_index.
    :param interval_kwargs: keyword-arguments for the interval_range function,
    specifying either the 'periods' -- the number of periods to generate, or
    'freq' -- the length of each interval. Optionally also specify 'closed' --
    whether the intervals are closed on the left, right, both or neither side.
    See pandas.interval_range for more details.

    :return: dataframe with brsm.admission_index as index. One column for each
    input column and interval. The values of the columns is the sum of
    True elements in the interval (admission_date - diagnosis_date) for each
    column in the input table. The columns are a multi-index, with the top
    level specifying the interval, as I#, where # is the number of the
    interval.
    """
    cols = list(mhs.filter(regex="(ICD_)|(OP_)|(ATC_)").columns)

    intervals = pd.interval_range(
        start=pd.Timedelta(0),
        end=pd.Timedelta('1825D'),
        **interval_kwargs
    )
    event_age = mhs.admission_date - mhs.event_date

    sums = []
    for interval in intervals:
        sums.append(
            mhs.loc[
                event_age.between(
                    interval.left,
                    interval.right,
                    inclusive=interval.closed
                ),
                ['admission_index'] + cols
            ]
            .groupby('admission_index')
            .sum()
        )

    res = pd.concat(
        sums,
        axis=1,
        keys=[f"I{x}" for x in range(len(intervals))]
    ).reindex(brsm.admission_index, fill_value=0)

    res.columns = list(map('_'.join, res.columns))
    return res


def group_icd_to_level(icd, level=None):
    if level == 'chapter':
        log.info("Rounding ICD codes to chapters")
        icd = icd.astype(int).rename(columns=round_icd_to_chapter)
        icd = pd.concat(
            [
                icd[[chapter]].sum(axis=1).rename(chapter)
                for chapter in icd_chapters
            ],
            axis=1,
        )

    return icd


def group_atc_to_level(atc, level=None):
    if level == 'full':
        return atc

    atc = atc.astype(int).rename(columns=atc_to_level_rounder(level))
    categories = sorted(atc.columns.unique())
    atc = pd.concat(
        [
            atc[[category]].sum(axis=1).rename(category)
            for category in categories
        ],
        axis=1
    )
    return atc


@cache
def summarize_patient_history(brsm, sources=None, num_icd=0, num_kva=0,
                              num_atc=0, intervals=None, icd_level=None,
                              atc_level=None):
    dfs = []
    if intervals is None:
        intervals = {'periods': 1}

    def stagger_and_sum(df, k):
        if k > 0:
            df = df.iloc[:, :k]
        df = stagger_events(df, brsm)
        df = sum_events_in_interval(df, brsm, **intervals)
        return df

    if sources:
        for source in sources:
            icd, op = multihot_icd_kva(source, brsm)
            if num_icd:
                icd = group_icd_to_level(icd, icd_level)
                icd.columns = [f'{source}_ICD_{col}' for col in icd.columns]
                dfs.append(stagger_and_sum(icd, num_icd))
            if num_kva:
                dfs.append(stagger_and_sum(op, num_kva))

    if num_atc:
        atc = multihot_atc(brsm)
        atc = group_atc_to_level(atc, atc_level)
        atc.columns = [f"ATC_{col}" for col in atc.columns]
        dfs.append(stagger_and_sum(atc, num_atc))

    log.info('Concatenating events')
    mh = pd.concat(dfs, axis=1, join='outer').fillna(0)

    return mh


def staggered_patient_history(brsm, sources=None, diagnoses=0, interventions=0,
                              meds=0):
    dfs = []

    def prune(data, k):
        if k > 0:
            data = data.iloc[:, :k]
        data = data[data.any(axis=1)]
        return data

    if sources:
        for source in sources:
            icd, op = multihot_icd_kva(source, brsm)
            if diagnoses:
                dfs.append(prune(icd, diagnoses))
            if interventions:
                dfs.append(prune(op, interventions))

    if meds:
        atc = multihot_atc(brsm)
        dfs.append(prune(atc, meds))

    log.info('Concatenating events')
    df = pd.concat(dfs, axis=1, join='outer').fillna(False)
    df = stagger_events(df, brsm)
    df['years_ago'] = (
        (df.admission_date - df.event_date)
        .dt.total_seconds() / (3600 * 24 * 365)
    )
    return df


def remove_events_outside_time_interval(
        events, index_visits, interval_days_start=0, interval_days_end=30):
    # events has multi-index (Alias, event_date)
    # index_visists has columns at least Alias, admission_date
    # Output has multi-index (Alias, KontaktId), and includes all events
    # that occurred within the specified time from the index. The same event
    # can be listed multiple times if it matches more than one visit.
    cols = list(events.columns) + ['Alias', 'KontaktId']
    events = (
        events
        .reset_index()
        .join(index_visits.set_index('Alias'), on='Alias', how='inner')
    )

    # Calculate the time between admission and diagnosis, in days
    dt = (events.event_date.dt.floor('1D') -
          events.admission_date.dt.floor('1D'))
    dt = dt.dt.total_seconds() / (3600 * 24)
    interval = (dt >= interval_days_start) & (dt <= interval_days_end)

    # Keep only the rows within the specified interval.
    return events.loc[interval, cols].set_index(['Alias', 'KontaktId'])


def make_mace(index):
    icd, op = _make_sos_codes('SV', index)
    events = remove_events_outside_time_interval(icd, index)
    mace = make_mace_table(index, icd_events=events, op_events=None)
    deaths = _make_death_in_30days(index)
    return mace.join(deaths)


def make_mace_table(index, icd_events, op_events):
    # This is still a bit WIP - haven't included all sources of events, or
    # intervention codes. Also, the list of ICD codes is different from
    # before.
    mace_codes_new = [
        "I200",
        "I21",
        "I22",
        "I26",  # Lungemboli
        "I441",
        "I442",
        "I46",
        "I470",
        "I472",
        "I490",
        "I71",  # Aortadissektion
        "J819",
    ]
    index = index.set_index(['Alias', 'KontaktId']).sort_index()
    diagnoses = pd.DataFrame(index=icd_events.index)
    mace_table = pd.DataFrame(index=index.index)
    for icd in mace_codes_new:
        diagnoses[icd] = icd_events.ICD.str.contains(icd)

    mace_table = mace_table.join(
        diagnoses.groupby('KontaktId')[mace_codes_new].any(),
        how='left'
    )
    return mace_table.fillna(False)


def _make_duplicated_key_mapping(key):
    # There are a few LopNr that maps to more than one Alias. I want to
    # remedy this by mapping all the Aliases for any LopNr to the last
    # used version. This is indicated by the column "SenPNr". This function
    # returns a dictionary that maps all the old Aliases to their last used
    # counterparts, for those that have multiple LopNr.
    dupes = (
        key[key.LopNr.duplicated(keep=False)]
        .sort_values(by=['LopNr', 'SenPNr'])
    )
    old = (
        dupes
        .drop_duplicates(subset=['LopNr'], keep='first')
        .set_index('LopNr')
        [['Alias']]
    )
    new = (
        dupes
        .drop_duplicates(subset=['LopNr'], keep='last')
        .set_index('LopNr')
        [['Alias']]
    )
    mapping = (
        old
        .join(new, lsuffix='_old')
        .set_index('Alias_old')
        .to_dict()
    )
    return mapping


def make_index():
    # Remaps Alias of duplicated LopNr to the last used Alias
    # Drops any visits with missing cause
    # For duplicated entries on the same patient, date and time, take the one
    # with the latest discharge date.
    # Does NOT drop missing hospitals (these are primarily Lund, Kristianstad
    # and Ystad, I think)
    # Does NOT drop visits with unknown discharge date
    key = read_csv(
        'SCB_Ekelund_LEV_Nyckel.csv',
        usecols=['Alias', 'LopNr', 'SenPNr']
    )
    alias_mapping = _make_duplicated_key_mapping(key)

    index_col_map = {
        'KontaktId': 'KontaktId',
        'Alias': 'Alias',
        'Sjukhus_Namn': 'hospital',
        'BesokOrsak_Kod': 'cause',
        'Vardkontakt_InskrivningDatum': 'admission_date',
        'Vardkontakt_UtskrivningDatum': 'discharge_date',
        'Patient_Kon': 'sex',
        'Vardkontakt_PatientAlderVidInskrivning': 'age'
    }
    index = (
        read_csv(
            'PATIENTLIGGAREN_Vårdkontakter_2017_2018.csv',
            usecols=index_col_map.keys(),
            parse_dates=[
                'Vardkontakt_InskrivningDatum',
                'Vardkontakt_UtskrivningDatum'
            ])
        .rename(columns=index_col_map)
        .dropna(subset=['cause'])
        .replace(alias_mapping)
        .join(key.set_index('Alias')['LopNr'], on='Alias', how='inner')
        .sort_values(
            by=['Alias', 'admission_date', 'discharge_date'],
            na_position='first'
        )
        .drop_duplicates(subset=['Alias', 'admission_date'], keep='last')
        .reset_index(drop=True)
    )
    return index


def _make_sos_codes(source, index):
    # WIP
    log.info(f'Extracting ICD and KVÅ events from {source}')

    source = source.upper()
    assert source in ['SV', 'OV']

    path = f'SOS_T_T_T_R_PAR_{source}_24129_2020.csv'

    log.info(f'Loading {path}')
    diagnosis_cols = ['hdia'] + [f'DIA{x}' for x in range(1, 31)]
    op_cols = ['OP']
    sos = (
        read_csv(
            path,
            usecols=['LopNr', 'INDATUM'] + diagnosis_cols + op_cols,
            parse_dates=['INDATUM'],
            low_memory=False)
        .rename(columns={'INDATUM': 'event_date'})
        .join(index.set_index('LopNr'), on='LopNr', how='inner')
        .set_index(['Alias', 'event_date'])
        .sort_index()
    )

    icd = (
        sos
        .loc[:, diagnosis_cols]
        .stack()
        .rename('ICD')
        .reset_index(level=2, drop=True)
        .reset_index()
        .drop_duplicates()
        .set_index(['Alias', 'event_date'])
    )
    # icd = pivot(icd)

    op = (
        sos
        .loc[:, 'OP']
        .str.split(' ')
        .explode()
        .rename('OP')
        .dropna()
        .reset_index()
        .drop_duplicates()
        .set_index(['Alias', 'event_date'])
    )

    # op = pivot(op)
    return icd, op


def _make_death_in_30days(index):
    deaths = read_csv(
        'SOS_R_DORS_24129_2020.csv',
        usecols=['LopNr', 'DODSDAT'],
        low_memory=False
    ).rename(columns={'DODSDAT': 'death_date'}).set_index('LopNr')

    deaths.death_date = pd.to_datetime(
        deaths.death_date.astype(str).apply(fix_dors_date),
        format='%Y%m%d'
    )

    deaths = index[['LopNr', 'Alias', 'KontaktId', 'admission_date']].join(
        deaths, on='LopNr'
    )
    dt = (deaths.death_date - deaths.admission_date
          ).dt.total_seconds() / (3600 * 24)

    # You're unlikely to have died before going to the hospital...
    # But it is possible that the day of death is unknown, at which point
    # date of death is rounded to the first of the month.
    deaths['death'] = (dt >= -30) & (dt <= 30)
    return deaths.set_index(['Alias', 'KontaktId'])[['death']]


@cache
def lisa(index, onehot=True):
    dfs = {year: read_lisa(year) for year in range(2012, 2019)}
    admission_year = index.admission_date.dt.year

    common_cols = [
        # "lisa_missing",
        "marital_status",
        "citizenship_eu15",
        "citizenship_eu28",
        "children_aged_0_3",
        "children_aged_4_6",
        "children_aged_7_10",
        "children_aged_11_15",
        "children_aged_16_17",
        "education_level_old",
        "education_level",
        "education_duration",
        "education_type",
        "education_focus",
        "graduation_decade",
        "occupational_status",
        "occupation_type",
        "occupation_code",
        "socioeconomic_group",
        "socioeconomic_class"
    ]

    yearly_cols = [
        'parental_benefit',
        'sickness_benefit',
        'received_sickness_benefit',
        'sickness_and_rehab_benefit',
        'sickness_benefit_days',
        'unemployment_benefit',
        'unemployment_days',
        'political_benefit',
        'early_retirement_benefit',
        'received_early_retirement_benefit',
        'sickness_pension_days',
        'capital_income',
        'retirement_pension',
        'social_benefit',
        'housing_benefit',
        'disposable_income_family_v1',
        'disposable_income_family_v2',
        'disposable_income',
        'lisa_missing'
    ]

    year_2017 = (
        index.loc[admission_year == 2017, ['LopNr']]
        .join(dfs[2016][common_cols], on='LopNr')
    )
    for year in range(2012, 2017):
        old_lisa = dfs[year][yearly_cols].rename(
            columns={x: f'{x}_{2017-year}_yrs_ago' for x in yearly_cols}
        )
        year_2017 = year_2017.join(old_lisa, on='LopNr')

    year_2018 = (
        index.loc[admission_year == 2018, ['LopNr']]
        .join(dfs[2017][common_cols], on='LopNr')
    )
    for year in range(2013, 2018):
        old_lisa = dfs[year][yearly_cols].rename(
            columns={x: f'{x}_{2018-year}_yrs_ago' for x in yearly_cols}
        )
        year_2018 = year_2018.join(old_lisa, on='LopNr')

    common = pd.concat([year_2017, year_2018], axis=0)
    indicator_missing = list(common.filter(like='missing'))
    common[indicator_missing] = common[indicator_missing].fillna(True)
    common = common.reindex(index.index)

    common = common.fillna({
        'marital_status': '*',
        'citizenship_eu15': 11,
        'citizenship_eu28': 11,
        'children_aged_0_3': -1,
        'children_aged_4_6': -1,
        'children_aged_7_10': -1,
        'children_aged_11_15': -1,
        'children_aged_16_17': -1,
        'education_level_old': '*',
        'education_level': 9,
        'education_duration': 9,
        'education_type': 9,
        'education_focus': '99',
        'graduation_decade': '***',
        'occupational_status': 6,
        'occupation_type': 0,
        'occupation_code': '*',
        'socioeconomic_group': '*',
        'socioeconomic_class': '*',
    })
    common = common.fillna(0).set_index('LopNr')

    if onehot:
        categorical_columns = [
            "marital_status",
            "citizenship_eu15",
            "citizenship_eu28",
            "children_aged_0_3",
            "children_aged_4_6",
            "children_aged_7_10",
            "children_aged_11_15",
            "children_aged_16_17",
            "education_level_old",
            "education_level",
            "education_duration",
            "education_type",
            "education_focus",
            "graduation_decade",
            "occupational_status",
            "occupation_type",
            "occupation_code",
            "socioeconomic_group",
            "socioeconomic_class",
        ]
        remaining_columns = common.columns.difference(categorical_columns)
        ohe = OneHotEncoder(sparse=False)
        return pd.concat([
            pd.DataFrame(
                ohe.fit_transform(common[categorical_columns]),
                index=common.index,
                columns=ohe.get_feature_names_out(),
            ),
            common[remaining_columns]
        ], axis=1).astype(float)
    else:
        return common


def read_lisa(year):
    assert year in range(2012, 2019)

    col_map = {
        'lopnr': 'LopNr',
        'Civil': 'marital_status',
        'MedbGrEg': 'citizenship_eu15',
        'MedbGrEg4': 'citizenship_eu28',
        'Barn0_3': 'children_aged_0_3',
        'Barn4_6': 'children_aged_4_6',
        'Barn7_10': 'children_aged_7_10',
        'Barn11_15': 'children_aged_11_15',
        'Barn16_17': 'children_aged_16_17',
        'Sun2000niva_old': 'education_level_old',
        'Sun2000niva': 'education_level_sun2000',
        'Sun2000Inr': 'education_focus',
        'ExamAr': 'graduation_year',
        'SyssStat11': 'occupational_status',
        'YrkStalln': 'occupation_type',
        # 'SsykStatus_J16': 'occupation_in_own_company'

        'ForLed': 'parental_benefit',
        'SjukPP': 'sickness_benefit',
        'SjukTyp': 'received_sickness_benefit',
        'SjukRe': 'sickness_and_rehab_benefit',
        'SjukP_Ndag_MiDAS': 'sickness_benefit_days',
        'ArbLos': 'unemployment_benefit',
        'ALosDag': 'unemployment_days',
        # Hur översätter man arbetsmarknadspolitiska åtgärder??
        'AmPol': 'political_benefit',
        'ForTid': 'early_retirement_benefit',
        'ForTidTyp': 'received_early_retirement_benefit',
        'SjukErs_Ndag_MiDAS': 'sickness_pension_days',
        'KapInk': 'capital_income',
        'SumAldP03': 'retirement_pension',
        'SocBidrFam': 'social_benefit',
        'BostBidrFam': 'housing_benefit',
        'DispInkKE': 'disposable_income_family_v1',
        'DispInkKE04': 'disposable_income_family_v2',
        'DispInk04': 'disposable_income'
    }

    if year > 2015:
        col_map['Ssyk4_2012_J16'] = 'occupation_code'

    if year == 2016:
        col_map['ESeG'] = 'socioeconomic_group'
    elif year >= 2017:
        col_map['ESeG_J16'] = 'socioeconomic_group'

    df = (
        read_csv(f"SCB_Ekelund_LEV_LISA{year}.csv", usecols=col_map.keys())
        .rename(columns=col_map)
        .set_index('LopNr')
    )

    # In the year 2016, for some reason, there are some nans in this column.
    # They should map to value 11, which is the designated "unknown"
    # citizenship code according to the LISA documentation.
    df.citizenship_eu15 = df.citizenship_eu15.fillna(11.0)
    df['lisa_missing'] = False

    day_cols = [
        'sickness_benefit_days',
        'unemployment_days',
        'sickness_pension_days'
    ]
    days = 366 if year in [2012, 2016] else 365
    df[day_cols] /= days

    sun2000 = df.education_level_sun2000.astype(str)
    df['education_level'] = sun2000.str[0].astype(int)
    df['education_duration'] = sun2000.str[1].astype(int)
    df['education_type'] = sun2000.str[2].astype(int)
    df['graduation_decade'] = df.graduation_year.str[:3]

    if year > 2015:
        df.education_focus = df.education_focus.str[:2]
        df.occupation_code = df.occupation_code.fillna('*').str[0]

        df.socioeconomic_group = (
            df.socioeconomic_group.fillna('*').astype(str).str[0]
        )
        df['socioeconomic_class'] = df.socioeconomic_group.map({
            '1': 'high',
            '2': 'high',
            '3': 'middle',
            '4': 'middle',
            '5': 'working',
            '6': 'working',
            '7': 'working',
            '*': '*'
        })

    return df
