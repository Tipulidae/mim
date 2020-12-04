import numpy as np

from mim.massage.carlson_ecg import (
    ECGStatus,
    status_of_ecg_dict,
    flatten_nested
)


def test_can_detect_missing_data_field():
    ecg = {
        'Patient': np.array([]),
        'Recording': np.array([]),
        'Measurements': np.array([])
    }
    assert ECGStatus.MISSING_DATA in status_of_ecg_dict(ecg)

    ecg['Data'] = np.array([])
    assert ECGStatus.MISSING_DATA not in status_of_ecg_dict(ecg)


def test_can_detect_missing_data_field_details():
    ecg = {
        'Patient': np.array([]),
        'Recording': np.array([]),
        'Measurements': np.array([]),
        'Data':
            np.array(
                ([], [], []),
                dtype=[('Foo', 'O'), ('Bar', 'O'), ('Baz', 'O')]
            )
    }

    assert ECGStatus.MISSING_LABELS in status_of_ecg_dict(ecg)
    assert ECGStatus.MISSING_ECG in status_of_ecg_dict(ecg)
    assert ECGStatus.MISSING_BEAT in status_of_ecg_dict(ecg)

    ecg['Data'] = np.array(
        ([], [], []),
        dtype=[('Labels', 'O'), ('Bar', 'O'), ('Baz', 'O')]
    )
    assert ECGStatus.MISSING_LABELS not in status_of_ecg_dict(ecg)
    assert ECGStatus.MISSING_ECG in status_of_ecg_dict(ecg)
    assert ECGStatus.MISSING_BEAT in status_of_ecg_dict(ecg)

    ecg['Data'] = np.array(
        ([], [], []),
        dtype=[('Labels', 'O'), ('ECG', 'O'), ('Baz', 'O')]
    )
    assert ECGStatus.MISSING_LABELS not in status_of_ecg_dict(ecg)
    assert ECGStatus.MISSING_ECG not in status_of_ecg_dict(ecg)
    assert ECGStatus.MISSING_BEAT in status_of_ecg_dict(ecg)

    ecg['Data'] = np.array(
        ([], [], []),
        dtype=[('Labels', 'O'), ('ECG', 'O'), ('Beat', 'O')]
    )
    assert ECGStatus.MISSING_LABELS not in status_of_ecg_dict(ecg)
    assert ECGStatus.MISSING_ECG not in status_of_ecg_dict(ecg)
    assert ECGStatus.MISSING_BEAT not in status_of_ecg_dict(ecg)


def test_can_detect_missing_measurements():
    ecg = {'Patient': [], 'Recording': []}
    assert ECGStatus.MISSING_MEASUREMENTS in status_of_ecg_dict(ecg)

    ecg['Measurements'] = np.array([])
    assert ECGStatus.MISSING_MEASUREMENTS not in status_of_ecg_dict(ecg)


def test_can_detect_missing_diagnoses():
    ecg = {
        'Patient': np.array([]),
        'Recording': np.array([]),
        'Measurements':
            np.array(
                [([[[]]], [[[]]], [[[]]])],
                dtype=[('Foo', 'O'), ('S', 'O'), ('R', 'O')]
            )
    }

    assert ECGStatus.MISSING_DIAGNOSES in status_of_ecg_dict(ecg)

    ecg['Measurements'] = np.array(
        [(np.array([[[['Hi'], ['There']]]]), [[[]]], [[[]]])],
        dtype=[('D', 'O'), ('S', 'O'), ('R', 'O')]
    )
    assert ECGStatus.MISSING_DIAGNOSES not in status_of_ecg_dict(ecg)


def test_flatten_ridiculous_non_empty_nesting():
    data = [[[[
        ["A fly can't bird"],
        ["But a bird can fly"]
    ]]]]

    expected = ["A fly can't bird", "But a bird can fly"]
    actual = flatten_nested(data)
    assert actual == expected


def test_flatten_ridiculous_empty_nesting():
    data = [[[
    ]]]

    expected = []
    actual = flatten_nested(data)
    assert actual == expected
