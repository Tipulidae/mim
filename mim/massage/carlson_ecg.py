from enum import Enum
from datetime import datetime

import numpy as np


class ECGStatus(Enum):
    MISSING_DATA = 0
    MISSING_LABELS = 1
    MISSING_ECG = 2
    MISSING_BEAT = 3
    MISSING_RHYTHM = 4
    MISSING_DIAGNOSES = 5
    MISSING_SUMMARY = 6
    MISSING_MEASUREMENTS = 7
    TECHNICAL_ERROR = 8
    BAD_DIAGNOSIS = 9
    FILE_MISSING = 10
    BAD_ECG_DIMENSIONS = 11
    BAD_BEAT_DIMENSIONS = 12
    EMPTY_ECG_ROWS = 13
    EMPTY_ECG_COLUMNS = 14
    EMPTY_BEAT_ROWS = 15
    EMPTY_BEAT_COLUMNS = 16
    MISSING_RECORDING = 17
    MISSING_FILE_FORMAT = 18
    MISSING_ID = 19
    MISSING_DATE = 20
    MISSING_DEVICE = 21
    MISSING_LEAD_SYSTEM = 22
    MISSING_LEAD_NAMES = 23
    BAD_LEAD_NAMES = 24
    BAD_DATE = 25
    MISSING_PATIENT = 26

    MISSING_GLASGOW = 27
    BAD_GLASGOW = 28


expected_lead_names = [
    'V1',
    'V2',
    'V3',
    'V4',
    'V5',
    'V6',
    'I',
    'II',
    'III',
    'aVR',
    'aVL',
    'aVF'
]

glasgow_vector_names = [
    'Ponset',
    'Pduration',
    'QRSonset',
    'QRSduration',
    'ST80amplitude',
    'Qduration',
    'Rduration',
    'Sduration',
    'Rprim_dur',
    'Sprim_dur',
    'Rbis_dur',
    'Sbis_dur',
    'STduration',
    'Tonset',
    'Tduration',
    'Ppos_dur',
    'Tpos_dur',
    'QRS_IntD',
    'Ppos_amp',
    'Pneg_amp',
    'pk2pkQRS_amp',
    'R1_amp',
    'Qamplitude',
    'Ramplitude',
    'Samplitude',
    'Rprim_amp',
    'Sprim_amp',
    'Rbis_amp',
    'Sbis_amp',
    'ST_amp',
    'STT28_amp',
    'STT38_amp',
    'Tpos_amp',
    'Tneg_amp',
    'QRSarea',
    'Parea',
    'Tarea',
    'Pmorphology',
    'Tmorphology',
    'Rnotch',
    'DeltaConf',
    'STslope',
    'QTinterval',
    'STM_amp',
    'ST60_amp',
    'STTmid_amp',
    'EndQRSnotch_amp',
    'PR_amp',
    'ST_ampAdj',
    'EndQRSnotchSlurOnset',
    'EndQRSnotchSlurPeak'
]

glasgow_scalar_names = [
    'QRSFrontalAxis',
    'PFrontalAxis',
    'STFrontalAxis',
    'TFrontalAxis',
    'QRSpvec48sv',
    'QRSpvec58sv',
    'QRSpvec68sv',
    'QRSpvec78sv',
    'QRSpvecMaxAmpl',
    'OverallPonset',
    'OverallPend',
    'OverallQRSonset',
    'OverallQRSend',
    'OverallTonset',
    'OverallTend',
    'HeartRateVariability',
    'StdDevNormRR',
    'LVHscore',
    'LVstrain',
    'OverallPdur',
    'OverallQRSdur',
    'OverallSTdur',
    'OverallTdur',
    'RmaxaVR',
    'RmaxaVL',
    'SampV1',
    'RampV5',
    'SampV1plusRampV5',
    'OverallPRint',
    'OverallQTint',
    'HeartRate',
    'PtermV1',
    'QTdisp',
    'QTc',
    'QTcHodge',
    'QTcBazett',
    'QTcFridericia',
    'QTcFramingham',
    'SinusRate',
    'SinusAveRR',
    'VentRate',
    'VentAveRR'
]


def ecg_status(ecg):
    return (
        data_status(ecg) |
        measurement_status(ecg) |
        patient_status(ecg) |
        recording_status(ecg)
    )


def data_status(ecg):
    if 'Data' not in ecg:
        return {ECGStatus.MISSING_DATA}

    data = ecg['Data']
    return (
        label_status(data) |
        raw_status(data) |
        beat_status(data) |
        lead_name_status(data)
    )


def measurement_status(ecg):
    if 'Measurements' not in ecg:
        return {ECGStatus.MISSING_MEASUREMENTS}

    measurements = ecg['Measurements']
    return (
        diagnose_status(measurements) |
        rhythm_status(measurements) |
        summary_status(measurements) |
        glasgow_status(measurements)
    )


def patient_status(ecg):
    if 'Patient' not in ecg:
        return {ECGStatus.MISSING_PATIENT}

    patient = ecg['Patient']

    alias = flatten_nested(extract_field(patient, 'ID'))
    if len(alias) == 0:
        return {ECGStatus.MISSING_ID}

    return set()


def recording_status(ecg):
    if 'Recording' not in ecg:
        return {ECGStatus.MISSING_RECORDING}

    recording = ecg['Recording']
    return (
        date_status(recording) |
        device_status(recording) |
        lead_system_status(recording)
    )


def label_status(data):
    if 'Labels' not in data.dtype.fields:
        return {ECGStatus.MISSING_LABELS}

    return set()


def raw_status(data):
    if 'ECG' not in data.dtype.fields:
        return {ECGStatus.MISSING_ECG}

    ecg = extract_field(data, 'ECG')[0][0]
    if is_ecg_shape_malformed(ecg.shape):
        return {ECGStatus.BAD_ECG_DIMENSIONS}

    status = set()
    ecg = ecg[:10000, :8]
    if empty_row_count(ecg) > 0:
        status.add(ECGStatus.EMPTY_ECG_ROWS)
    if empty_row_count(ecg.T) > 0:
        status.add(ECGStatus.EMPTY_ECG_COLUMNS)

    return status


def beat_status(data):
    if 'Beat' not in data.dtype.fields:
        return {ECGStatus.MISSING_BEAT}

    beat = extract_field(data, 'Beat')[0][0]
    if is_beat_shape_malformed(beat.shape):
        return {ECGStatus.BAD_BEAT_DIMENSIONS}

    status = set()
    beat = beat[:1200, :8]
    if empty_row_count(beat) > 0:
        status.add(ECGStatus.EMPTY_BEAT_ROWS)
    if empty_row_count(beat.T) > 0:
        status.add(ECGStatus.EMPTY_BEAT_COLUMNS)

    return status


def diagnose_status(measurements):
    diagnoses = flatten_nested(extract_field(measurements, 'D'))
    if len(diagnoses) == 0:
        return {ECGStatus.MISSING_DIAGNOSES}

    if any(is_bad_diagnose(d) for d in diagnoses):
        return {ECGStatus.BAD_DIAGNOSIS}

    return set()


def rhythm_status(measurements):
    rhythms = flatten_nested(extract_field(measurements, 'R'))
    if len(rhythms) == 0:
        return {ECGStatus.MISSING_RHYTHM}

    return set()


def summary_status(measurements):
    summary = flatten_nested(extract_field(measurements, 'S'))
    if len(summary) == 0:
        return {ECGStatus.MISSING_SUMMARY}

    if 'Technical error' in summary:
        return {ECGStatus.TECHNICAL_ERROR}

    return set()


def glasgow_status(measurements):
    try:
        glasgow_data = measurements[0][0]
    except IndexError:
        return {ECGStatus.MISSING_GLASGOW}

    return (
        glasgow_scalar_status(glasgow_data) |
        glasgow_vector_status(glasgow_data)
    )


def glasgow_scalar_status(glasgow_data):
    all_names = glasgow_data.dtype.names
    for name in glasgow_scalar_names:
        if name not in all_names:
            return {ECGStatus.MISSING_GLASGOW}

    for name in glasgow_scalar_names:
        if len(glasgow_data[name][0]) != 1:
            return {ECGStatus.BAD_GLASGOW}

    return set()


def glasgow_vector_status(glasgow_data):
    all_names = glasgow_data.dtype.names
    for name in glasgow_vector_names:
        if name not in all_names:
            return {ECGStatus.MISSING_GLASGOW}

    for name in glasgow_vector_names:
        if len(glasgow_data[name][0]) != 12:
            return {ECGStatus.BAD_GLASGOW}

    return set()


def date_status(recording):
    try:
        date = extract_field(recording, 'Date')[0][0][0]
    except IndexError:
        return {ECGStatus.MISSING_DATE}

    try:
        datetime.strptime(date, "%d-%b-%Y %H:%M:%S")
    except ValueError:
        return {ECGStatus.BAD_DATE}

    return set()


def device_status(recording):
    try:
        extract_field(recording, 'Device')[0][0][0]
    except IndexError:
        return {ECGStatus.MISSING_DEVICE}

    return set()


def lead_system_status(recording):
    try:
        extract_field(recording, 'Lead_system')[0][0][0]
    except IndexError:
        return {ECGStatus.MISSING_LEAD_SYSTEM}

    return set()


def lead_name_status(data):
    try:
        labels = extract_field(data, 'Labels')[0][0][0]
        actual_lead_names = [x[0] for x in labels]
    except IndexError:
        return {ECGStatus.MISSING_LEAD_NAMES}

    if actual_lead_names != expected_lead_names:
        return {ECGStatus.BAD_LEAD_NAMES}

    return set()


def extract_field(data, field):
    # Given some data, check that it's a numpy-array with a field
    # with the given name, then return the contents of that field, otherwise
    # return an empty list.
    if not isinstance(data, np.ndarray):
        return []

    dtypes = data.dtype.fields
    if dtypes is None:
        return []

    if field not in dtypes:
        return []

    return data[field]


def is_empty_nested(data):
    return len(data[0][0]) == 0


def flatten_nested(data):
    if len(data) == 0:
        return []
    if is_empty_nested(data):
        return []
    else:
        return [x[0] for x in data[0][0][0]]


def contains_bad_diagnosis(diagnoses):
    for diagnose in diagnoses:
        if is_bad_diagnose(diagnose):
            return True
    return False


def is_bad_diagnose(diagnose):
    return (
        diagnose.startswith("Lead(s) unsuitable for analysis:") or
        diagnose in bad_diagnoses
    )


bad_diagnoses = {
    '--- No further analysis made ---',
    '--- Possible arm lead reversal - hence only aVF, V1-V6 analyzed ---',
    '--- Possible arm/leg lead interchange - hence only V1-V6 analyzed ---',
    '--- Possible limb lead reversal - hence only V1-V6 analyzed ---',
    '--- Possible measurement error ---',
    '--- Similar QRS in V leads ---',
    '--- Suggests dextrocardia ---',
    '--- Technically unsatisfactory tracing ---'
    'IV conduction defect',
    'Possible faulty V2 - omitted from analysis',
    'Possible faulty V3 - omitted from analysis',
    'Possible faulty V4 - omitted from analysis',
    'Possible faulty V5 - omitted from analysis',
    'Possible sequence error: V1,V2 omitted',
    'Possible sequence error: V2,V3 omitted',
    'Possible sequence error: V3,V4 omitted',
    'Possible sequence error: V4,V5 omitted',
    'V1/V2 are at least one interspace too high and have been omitted from '
    'the analysis'
}


def is_ecg_shape_malformed(shape):
    return (
        len(shape) != 2 or
        shape[0] < 10000 or
        shape[1] < 8
    )


def is_beat_shape_malformed(shape):
    return (
        len(shape) != 2 or
        shape[0] < 1200 or
        shape[1] < 8
    )


def empty_row_count(x):
    return len(x) - x.any(axis=1).sum()
