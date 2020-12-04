from enum import Enum


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


def status_of_ecg_dict(ecg):
    status = set()
    if 'Data' not in ecg:
        status.add(ECGStatus.MISSING_DATA)
    elif dtypes := ecg['Data'].dtype.fields:
        if 'Labels' not in dtypes:
            status.add(ECGStatus.MISSING_LABELS)
        else:
            # TODO: Check that labels are good
            pass

        if 'ECG' not in dtypes:
            status.add(ECGStatus.MISSING_ECG)
        else:
            # TODO: Check dimensions of ECG
            # TODO: Check for empty columns and rows
            pass

        if 'Beat' not in dtypes:
            status.add(ECGStatus.MISSING_BEAT)
        else:
            # TODO: Check dimensions of Beat
            # TODO: Check for empty columns and rows
            pass

    if 'Measurements' not in ecg:
        status.add(ECGStatus.MISSING_MEASUREMENTS)
    elif dtypes := ecg['Measurements'].dtype.fields:
        if 'D' not in dtypes or is_empty_nested(ecg['Measurements']['D']):
            status.add(ECGStatus.MISSING_DIAGNOSES)
        else:
            diagnoses = flatten_nested(ecg['Measurements']['D'])
            if contains_bad_diagnosis(diagnoses):
                status.add(ECGStatus.BAD_DIAGNOSIS)

        if 'R' not in dtypes or is_empty_nested(ecg['Measurements']['R']):
            status.add(ECGStatus.MISSING_RHYTHM)
        if 'S' not in dtypes or is_empty_nested(ecg['Measurements']['S']):
            status.add(ECGStatus.MISSING_SUMMARY)
        else:
            if 'Technical error' in flatten_nested(ecg['Measurements']['S']):
                status.add(ECGStatus.TECHNICAL_ERROR)

    return status


def is_empty_nested(data):
    return len(data[0][0]) == 0


def flatten_nested(data):
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
