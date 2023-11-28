import os

import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from projects.physionet23 import physionet_helper_code as helper

DATA_PATH = '/scratch/puck/axel/physionet_2023/training'
EEG_LEADS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz'
]


def load_patient_metadata(patient_id):
    patient_metadata_file = os.path.join(
        DATA_PATH, patient_id, patient_id + '.txt')

    return helper.load_text_file(patient_metadata_file)


def load_record_metadata(patient_id):
    recording_metadata_file = os.path.join(
        DATA_PATH, patient_id, patient_id + '.tsv')
    return helper.load_text_file(recording_metadata_file)


def make_index():
    patient_ids = helper.find_data_folders(DATA_PATH)
    index = pd.DataFrame(index=pd.Index(patient_ids, name='PatientID'))
    return index


def make_labels(index):
    def outcome_from_single_patient(pid):
        md = load_patient_metadata(pid)
        outcome = helper.get_outcome(md)
        cpc = helper.get_cpc(md)
        return [outcome, cpc]
    return pd.DataFrame(
        map(outcome_from_single_patient, index.index),
        columns=['outcome', 'cpc'],
        index=index.index
    )


def make_patient_features(index):
    def features_from_single_patient(pid):
        md = load_patient_metadata(pid)
        return pd.Series(dict(
            age=helper.get_age(md),
            sex=helper.get_sex(md),
            rosc=helper.get_rosc(md),
            ohca=helper.get_ohca(md),
            vfib=helper.get_vfib(md),
            ttm=helper.get_ttm(md)
        ), name=pid)
    x = pd.DataFrame(
        map(features_from_single_patient, index.index),
        index=index.index
    )

    def handle_missing_data(column, fill_value):
        x[f'{column}_unknown'] = x[column].isna().astype(float)
        x.loc[:, column] = x.loc[:, column].fillna(fill_value).astype(float)

    handle_missing_data('ohca', 0.0)
    handle_missing_data('vfib', 0.0)
    handle_missing_data('rosc', 0.0)
    handle_missing_data('ttm', 0.0)
    handle_missing_data('age', 0.0)
    return x.join(one_hot_encode(x[['sex']])).drop(columns=['sex'])


def one_hot_encode(df):
    assert len(df.columns) == 1
    name = df.columns[0]
    df = df.fillna('Other')
    ohe = OneHotEncoder(sparse_output=False, dtype=float)
    ohe.fit(df)
    new_df = pd.DataFrame(
        ohe.transform(df),
        index=df.index,
        columns=[f'{name}_{c}'.lower() for c in ohe.categories_[0]]
    )
    return new_df


def load_eegs(pid):
    recordings = list()
    recording_metadata = load_record_metadata(pid)
    recording_ids = helper.get_recording_ids(recording_metadata)
    missing = []
    for recording_id in recording_ids:
        if recording_id != 'nan':
            recording_location = os.path.join(
                DATA_PATH, pid, recording_id)
            recording_data, _, channels = helper.load_recording(
                recording_location)
            recording_data = helper.reorder_recording_channels(
                recording_data, channels, EEG_LEADS)
            missing.append(False)
        else:
            recording_data = np.zeros((18, 30000))
            missing.append(True)

        recordings.append(recording_data)

    return np.stack(recordings)


def extract_brain_waves_from_patient(pid, which='all', clip=200, **settings):
    all_eegs = load_eegs(pid)

    def extract(eeg):
        return extract_brain_waves_from_eeg(eeg, **settings)

    if which == 'all':
        eegs = all_eegs
    elif which == 'last':
        eegs = all_eegs[[-1]]
    elif isinstance(which, list):
        eegs = all_eegs[which]
    else:
        raise ValueError('Invalid option for which EEGs to use: must be '
                         'either None, "last", or list of ints.')

    eegs = np.concatenate(list(map(extract, eegs)))
    if clip > 0:
        return eegs.clip(-clip, clip) / 100
    else:
        return eegs


def extract_brain_waves_from_eeg(
        eeg, mean=False, alpha=True, beta=True, delta=True, theta=True):
    def welch(fmin, fmax):
        return mne.time_frequency.psd_array_welch(
            eeg, sfreq=100, fmin=fmin, fmax=fmax, verbose=False)[0]

    waves = []
    if alpha:
        waves.append(welch(8.0, 12.0))
    if beta:
        waves.append(welch(12.0, 30.0))
    if delta:
        waves.append(welch(0.5, 4.0))
    if theta:
        waves.append(welch(4.0, 8.0))

    if mean:
        return np.concatenate(list(map(lambda x: x.mean(axis=1), waves)))
    else:
        return np.hstack(waves).flatten()
