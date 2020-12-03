from enum import Enum


class ECGStatus(Enum):
    MISSING_FIELD_DATA = 0
    MISSING_FIELD_LABELS = 1
    MISSING_FIELD_ECG = 2
    MISSING_FIELD_BEAT = 3


def status_of_ecg_dict(ecg):
    status = set()
    if 'Data' not in ecg:
        status.add(ECGStatus.MISSING_FIELD_DATA)
    else:
        if 'Labels' not in ecg['Data']:
            status.add(ECGStatus.MISSING_FIELD_LABELS)
        if 'ECG' not in ecg['Data']:
            status.add(ECGStatus.MISSING_FIELD_ECG)
        if 'Beat' not in ecg['Data']:
            status.add(ECGStatus.MISSING_FIELD_BEAT)

    return status
