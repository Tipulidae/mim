import base64
import glob
import zlib
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from scipy.signal import resample
# trunk-ignore(bandit/B410)
from lxml import etree

from mim.util.metadata import Metadata
from .carlson_ecg import empty_row_count


@dataclass
class ECGStatus:
    missing_patient_demographics: bool = False
    missing_test_demographics: bool = False
    missing_alias: bool = False
    missing_age: bool = False
    missing_sex: bool = False
    missing_height: bool = False
    missing_weight: bool = False
    missing_location: bool = False
    missing_date: bool = False
    missing_time: bool = False
    bad_date: bool = False

    missing_median: bool = False
    missing_median_lead: bool = False
    empty_median_rows: bool = False
    empty_median_columns: bool = False
    bad_median_crc: bool = False
    median_resampled: bool = False

    missing_rhythm: bool = False
    missing_rhythm_lead: bool = False
    empty_rhythm_rows: bool = False
    empty_rhythm_columns: bool = False
    bad_rhythm_crc: bool = False
    rhythm_resampled: bool = False


expected_lead_names = [
    'V1',
    'V2',
    'V3',
    'V4',
    'V5',
    'V6',
    'I',
    'II'
]


class MuseECG:
    def __init__(self, path):
        parser = etree.XMLParser(encoding="iso-8859-1")
        self.path = path
        # trunk-ignore(bandit/B320)
        self.tree = etree.parse(path, parser)
        self.status = ECGStatus()

    def demographics(self):
        patient_demo = self.tree.find('PatientDemographics')
        if patient_demo is None:
            self.status.missing_patient_demographics = True

        test_demo = self.tree.find('TestDemographics')
        if test_demo is None:
            self.status.missing_test_demographics = True

        def get_attribute(element, name, dtype):
            x = element.find(name)
            if dtype == 'string':
                return get_string(x)
            else:
                return get_int(x)

        def get_string(x):
            return '' if x is None else x.text

        def get_int(x):
            return 0 if x is None else int(x.text)

        def get_date():
            if test_demo is None:
                self.status.missing_date = True
                return "NaT"

            date = test_demo.find('AcquisitionDate').text
            time = test_demo.find('AcquisitionTime').text

            if date is None:
                self.status.missing_date = True
                return "NaT"
            if time is None:
                self.status.missing_time = True
                time = ''
            timestamp = pd.to_datetime(
                f"{date} {time}",
                format="%m-%d-%Y %H:%M:%S",
                errors='coerce'
            )
            if timestamp is pd.NaT:
                self.status.bad_date = True
                return "NaT"

            return timestamp.strftime("%Y-%m-%d %H:%M:%S")

        demographics = {
            'Alias': get_attribute(patient_demo, 'PatientID', 'string'),
            'age': get_attribute(patient_demo, 'PatientAge', 'int'),
            'sex': get_attribute(patient_demo, 'Gender', 'string'),
            'height': get_attribute(patient_demo, 'HeightCM', 'int'),
            'weight': get_attribute(patient_demo, 'WeightKG', 'int'),
            'location': get_attribute(test_demo, 'LocationName', 'string'),
            'date': get_date()
        }
        if demographics['Alias'] == '':
            self.status.missing_alias = True
        if demographics['age'] == 0:
            self.status.missing_age = True
        if demographics['sex'] == '':
            self.status.missing_sex = True
        if demographics['height'] == 0:
            self.status.missing_height = True
        if demographics['weight'] == 0:
            self.status.missing_weight = True
        if demographics['location'] == '':
            self.status.missing_location = True

        return demographics

    def _parse_waveform(self, waveform_type):
        assert waveform_type in ['Median', 'Rhythm']

        size = 600 if waveform_type == 'Median' else 5000
        elements = self.tree.xpath(
            f"//Waveform[contains(., '{waveform_type}')]"
        )
        if elements:
            element = elements[0]
        else:
            if waveform_type == 'Median':
                self.status.missing_median = True
            else:
                self.status.missing_rhythm = True
            return np.zeros((size, 8), np.int16)

        element_dict = {
            element.find('LeadID').text: element
            for element in element.findall('LeadData')
        }

        waveform = np.stack(
            [self.parse_waveform_data(element_dict, lead, size, waveform_type)
             for lead in expected_lead_names],
            axis=-1
        )
        if empty_row_count(waveform) > 0:
            if waveform_type == 'Median':
                self.status.empty_median_rows = True
            else:
                self.status.empty_rhythm_rows = True

        if empty_row_count(waveform.T) > 0:
            if waveform_type == 'Median':
                self.status.empty_median_columns = True
            else:
                self.status.empty_rhythm_columns = True

        return waveform

    def parse_waveform_data(self, element_dict, lead, size, waveform_type):
        if lead not in element_dict:
            if waveform_type == 'Median':
                self.status.missing_median_lead = True
            else:
                self.status.missing_rhythm_lead = True

            return np.zeros((size,), np.int16)

        element = element_dict[lead]
        waveform_element = element.find('WaveFormData')
        if waveform_element is None:
            if waveform_type == 'Median':
                self.status.missing_median_lead = True
            else:
                self.status.missing_rhythm_lead = True

            return np.zeros((size,), np.int16)

        expected_crc = element.find('LeadDataCRC32').text
        data = base64.b64decode(waveform_element.text)
        if str(zlib.crc32(data)) != expected_crc:
            if waveform_type == 'Median':
                self.status.bad_median_crc = True
            else:
                self.status.bad_rhythm_crc = True

        data = np.frombuffer(data, dtype=np.int16)

        if len(data) != size:
            if waveform_type == 'Median':
                self.status.median_resampled = True
            else:
                self.status.rhythm_resampled = True
            data = resample(data, size).astype(np.int16)

        return data

    def median(self):
        return self._parse_waveform('Median')

    def rhythm(self):
        return self._parse_waveform('Rhythm')


def create_hdf5_sk1718():
    print('globbing...')
    xml_paths = list(sorted(
        glob.iglob(
            '/tank/air-crypt/legacy/air-crypt-esc-trop/andersb/sk1718-ECG/'
            'decomp/MIL*/MIL/*.xml',
            recursive=True
        )
    ))
    print('Making hdf5 file...')
    to_hdf5(xml_paths, '/tank/air-crypt/axel/sk1718_ecg.hdf5')


def to_hdf5(ecg_paths, target_path):
    status_keys = list(sorted(asdict(ECGStatus()).keys()))

    def get_status(status):
        status_dict = asdict(status)
        return [status_dict[key] for key in status_keys]

    with h5py.File(target_path, 'w') as data:
        n = len(ecg_paths)
        data.create_dataset(
            "raw",
            (n, 5000, 8),
            dtype='int16',
            chunks=(1, 5000, 8),
            fletcher32=True
        )
        data.create_dataset(
            "beat",
            (n, 600, 8),
            dtype='int16',
            chunks=(1, 600, 8),
            fletcher32=True
        )

        meta = data.create_group('meta')
        meta.create_dataset(
            "date",
            (n,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        meta.create_dataset(
            "Alias",
            (n,),
            dtype='int64'
        )
        meta.create_dataset(
            "age",
            (n,),
            dtype='int64'
        )
        meta.create_dataset(
            "sex",
            (n,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        meta.create_dataset(
            "path",
            (n,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        meta.create_dataset(
            "location",
            (n,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        meta.create_dataset(
            "status",
            (n, len(status_keys)),
            dtype=bool
        )
        meta.create_dataset(
            "status_keys",
            (len(status_keys),),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        meta.create_dataset(
            "lead_names",
            (len(expected_lead_names),),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        meta.create_dataset(
           "report",
           (1,),
           dtype=h5py.string_dtype(encoding='utf-8')
        )

        data['meta']['status_keys'][()] = status_keys
        data['meta']['path'][()] = ecg_paths
        data['meta']['lead_names'][()] = expected_lead_names
        data['meta']['report'][()] = Metadata().report(
            conda=False, file_data=False, as_string=True)

        for i, path in tqdm(enumerate(ecg_paths), total=n):
            ecg = MuseECG(path)
            data['raw'][i] = ecg.rhythm()
            data['beat'][i] = ecg.median()

            demographics = ecg.demographics()
            data['meta']['date'][i] = demographics['date']
            data['meta']['Alias'][i] = int(demographics['Alias'])
            data['meta']['location'][i] = demographics['location']
            data['meta']['status'][i] = get_status(ecg.status)
            data['meta']['age'][i] = demographics['age']
            data['meta']['sex'][i] = demographics['sex']
