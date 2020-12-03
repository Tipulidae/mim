from mim.massage.carlson_ecg import ECGStatus, status_of_ecg_dict


def test_can_detect_missing_data_field():
    ecg = {'Patient': [], 'Recording': [], 'Measurements': []}
    assert ECGStatus.MISSING_FIELD_DATA in status_of_ecg_dict(ecg)

    ecg['Data'] = []
    assert ECGStatus.MISSING_FIELD_DATA not in status_of_ecg_dict(ecg)


def test_can_detect_missing_data_field_details():
    ecg = {
        'Patient': [],
        'Recording': [],
        'Measurements': [],
        'Data': {
            'Foo': [],
            'Bar': []
        }
    }
    assert ECGStatus.MISSING_FIELD_LABELS in status_of_ecg_dict(ecg)
    assert ECGStatus.MISSING_FIELD_ECG in status_of_ecg_dict(ecg)
    assert ECGStatus.MISSING_FIELD_BEAT in status_of_ecg_dict(ecg)

    ecg['Data']['Labels'] = []
    assert ECGStatus.MISSING_FIELD_LABELS not in status_of_ecg_dict(ecg)
    assert ECGStatus.MISSING_FIELD_ECG in status_of_ecg_dict(ecg)
    assert ECGStatus.MISSING_FIELD_BEAT in status_of_ecg_dict(ecg)

    ecg['Data']['ECG'] = []
    assert ECGStatus.MISSING_FIELD_LABELS not in status_of_ecg_dict(ecg)
    assert ECGStatus.MISSING_FIELD_ECG not in status_of_ecg_dict(ecg)
    assert ECGStatus.MISSING_FIELD_BEAT in status_of_ecg_dict(ecg)

    ecg['Data']['Beat'] = []
    assert ECGStatus.MISSING_FIELD_LABELS not in status_of_ecg_dict(ecg)
    assert ECGStatus.MISSING_FIELD_ECG not in status_of_ecg_dict(ecg)
    assert ECGStatus.MISSING_FIELD_BEAT not in status_of_ecg_dict(ecg)
