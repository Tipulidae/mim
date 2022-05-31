import glob

import scipy.io as sio
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from mim.util.metadata import Metadata
from mim.massage.carlson_ecg import (
    ECGStatus,
    ecg_status,
    expected_lead_names,
    glasgow_scalar_names,
    glasgow_vector_names,
    glasgow_diagnoses_index,
    glasgow_diagnoses,
    glasgow_rhythms_index,
    glasgow_rhythms,
    extract_field,
    flatten_nested
)


class ECG:
    def __init__(self, path):
        self.path = path
        self.ecg_dict = sio.loadmat(path)
        self.status = ecg_status(self.ecg_dict)

        self.raw = self.init_raw()
        self.beat = self.init_beat()
        self.metadata = self.init_metadata()

    def init_raw(self):
        if self.is_raw_ecg_ok:
            return self.ecg_dict['Data']['ECG'][0][0][:10000, :8]
        else:
            return np.zeros((10000, 8))

    def init_beat(self):
        if self.is_beat_ecg_ok:
            return self.ecg_dict['Data']['Beat'][0][0][:1200, :8]
        else:
            return np.zeros((1200, 8))

    def init_metadata(self):
        meta = dict()
        meta['date'] = self.init_date()
        meta['alias'] = self.init_alias()
        meta['path'] = self.path
        meta['device'] = self.init_device()
        meta['lead_system'] = self.init_lead_system()
        meta['status'] = [x in self.status for x in ECGStatus]
        return meta

    def init_date(self):
        if self.is_date_ok:
            date_string = self.ecg_dict['Recording']['Date'][0][0][0]
            timestamp = pd.to_datetime(date_string, format="%d-%b-%Y %H:%M:%S")
            return timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return "NaT"

    def init_alias(self):
        if self.is_alias_ok:
            return self.ecg_dict['Patient']['ID'][0][0][0]
        else:
            return ""

    def init_device(self):
        if self.is_device_ok:
            return extract_field(self.ecg_dict['Recording'], 'Device')[0][0][0]
            # return self.ecg_dict['Recording']['Device'][0][0][0]
        else:
            return "Unknown device"

    def init_lead_system(self):
        if self.is_lead_system_ok:
            return self.ecg_dict['Recording']['Lead_system'][0][0][0]
        else:
            return "Unknown lead system"

    def init_glasgow_vectors(self):
        if self.is_glasgow_ok:
            return np.vstack(
                [self.ecg_dict['Measurements'][0][0][x][0]
                 for x in glasgow_vector_names]
            )
        else:
            return np.zeros((len(glasgow_vector_names), 12))

    def init_glasgow_scalars(self):
        if self.is_glasgow_ok:
            return np.array(
                [self.ecg_dict['Measurements'][0][0][x][0][0]
                 for x in glasgow_scalar_names]
            )
        else:
            return np.zeros(len(glasgow_scalar_names))

    def init_glasgow_diagnoses(self):
        diagnoses_vector = np.zeros(
            (len(glasgow_diagnoses_index),)).astype(bool)
        if ECGStatus.MISSING_DIAGNOSES not in self.status:
            diagnoses = flatten_nested(extract_field(
                self.ecg_dict['Measurements'], 'D'))

            for d in diagnoses:
                if d in glasgow_diagnoses_index:
                    diagnoses_vector[glasgow_diagnoses_index[d]] = True

        return diagnoses_vector

    def init_glasgow_rhythms(self):
        rhythm_vector = np.zeros((len(glasgow_rhythms_index),)).astype(bool)

        if ECGStatus.MISSING_RHYTHM in self.status:
            return rhythm_vector

        rhythms = flatten_nested(extract_field(
            self.ecg_dict['Measurements'], 'R'
        ))
        if not rhythms:
            return rhythm_vector

        parts = set.union(*[set(r.split(' with ')) for r in rhythms])
        for part in parts:
            if part in glasgow_rhythms_index:
                rhythm_vector[glasgow_rhythms_index[part]] = True

        return rhythm_vector

    @property
    def is_raw_ecg_ok(self):
        return self.status.isdisjoint({
            ECGStatus.MISSING_DATA,
            ECGStatus.MISSING_ECG,
            ECGStatus.BAD_ECG_DIMENSIONS
        })

    @property
    def is_beat_ecg_ok(self):
        return self.status.isdisjoint({
            ECGStatus.MISSING_DATA,
            ECGStatus.MISSING_BEAT,
            ECGStatus.BAD_BEAT_DIMENSIONS
        })

    @property
    def is_date_ok(self):
        return self.status.isdisjoint({
            ECGStatus.MISSING_RECORDING,
            ECGStatus.MISSING_DATE,
            ECGStatus.BAD_DATE
        })

    @property
    def is_alias_ok(self):
        return self.status.isdisjoint({
            ECGStatus.MISSING_PATIENT,
            ECGStatus.MISSING_ID
        })

    @property
    def is_device_ok(self):
        return self.status.isdisjoint({
            ECGStatus.MISSING_RECORDING,
            ECGStatus.MISSING_DEVICE
        })

    @property
    def is_lead_system_ok(self):
        return self.status.isdisjoint({
            ECGStatus.MISSING_RECORDING,
            ECGStatus.MISSING_LEAD_SYSTEM,
            ECGStatus.MISSING_LEAD_NAMES,
            ECGStatus.BAD_LEAD_NAMES
        })

    @property
    def is_glasgow_ok(self):
        return self.status.isdisjoint({
            ECGStatus.MISSING_MEASUREMENTS,
            ECGStatus.MISSING_GLASGOW,
            ECGStatus.BAD_GLASGOW
        })


def create_esc_trop_ecg_hdf5():
    esc_trop_paths = list(sorted(
        glob.iglob('/mnt/air-crypt/air-crypt-esc-trop/ekg/mat/**/*.mat',
                   recursive=True)
    ))
    to_hdf5(esc_trop_paths, '/mnt/air-crypt/air-crypt-esc-trop/axel/ecg.hdf5')


def to_hdf5(ecg_paths, target_path):
    with h5py.File(target_path, 'w') as data:
        n = len(ecg_paths)
        data.create_dataset(
            "raw",
            (n, 10000, 8),
            chunks=(1, 10000, 8),
            fletcher32=True
        )
        data.create_dataset(
            "beat",
            (n, 1200, 8),
            chunks=(1, 1200, 8),
            fletcher32=True
        )

        glasgow = data.create_group('glasgow')
        glasgow.create_dataset(
            'vectors',
            (n, len(glasgow_vector_names), 12),
            chunks=(1, len(glasgow_vector_names), 12),
            fletcher32=True
        )
        glasgow.create_dataset(
            'scalars',
            (n, len(glasgow_scalar_names)),
            chunks=(1, len(glasgow_scalar_names)),
            fletcher32=True
        )
        glasgow.create_dataset(
            'diagnoses',
            (n, len(glasgow_diagnoses_index)),
            dtype=np.bool,
            chunks=(1, len(glasgow_diagnoses_index)),
            fletcher32=True
        )
        glasgow.create_dataset(
            'rhythms',
            (n, len(glasgow_rhythms_index)),
            dtype=np.bool,
            chunks=(1, len(glasgow_rhythms_index)),
            fletcher32=True
        )

        meta = data.create_group('meta')
        meta.create_dataset(
            "date",
            (n,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        meta.create_dataset(
            "alias",
            (n,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        meta.create_dataset(
            "path",
            (n,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        meta.create_dataset(
            "device",
            (n,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        meta.create_dataset(
            "lead_system",
            (n,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        meta.create_dataset(
            "status",
            (n, len(ECGStatus)),
            dtype=np.bool
        )
        meta.create_dataset(
            "status_keys",
            (len(ECGStatus),),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        meta.create_dataset(
            "glasgow_vector_names",
            (len(glasgow_vector_names),),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        meta.create_dataset(
            "glasgow_scalar_names",
            (len(glasgow_scalar_names),),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        meta.create_dataset(
            "glasgow_diagnoses_names",
            (len(glasgow_diagnoses_index),),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        meta.create_dataset(
            "glasgow_rhythms_names",
            (len(glasgow_rhythms_index),),
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

        data['meta']['status_keys'][()] = [x.name for x in ECGStatus]
        data['meta']['path'][()] = ecg_paths
        data['meta']['glasgow_vector_names'][()] = glasgow_vector_names
        data['meta']['glasgow_scalar_names'][()] = glasgow_scalar_names
        data['meta']['glasgow_diagnoses_names'][()] = glasgow_diagnoses
        data['meta']['glasgow_rhythms_names'][()] = glasgow_rhythms
        data['meta']['lead_names'][()] = expected_lead_names
        data['meta']['report'][()] = Metadata().report(
            conda=False, file_data=False, as_string=True)

        for i, path in tqdm(enumerate(ecg_paths), total=n):
            ecg = ECG(path)
            data['raw'][i] = ecg.raw
            data['beat'][i] = ecg.beat
            data['glasgow']['scalars'][i] = ecg.init_glasgow_scalars()
            data['glasgow']['vectors'][i] = ecg.init_glasgow_vectors()
            data['glasgow']['diagnoses'][i] = ecg.init_glasgow_diagnoses()
            data['glasgow']['rhythms'][i] = ecg.init_glasgow_rhythms()
            meta = ecg.metadata
            data['meta']['date'][i] = meta['date']
            data['meta']['alias'][i] = meta['alias']
            data['meta']['device'][i] = meta['device']
            data['meta']['lead_system'][i] = meta['lead_system']
            data['meta']['status'][i] = meta['status']


def create_ecg_dataframe_from_hdf5(path):
    with h5py.File(path, 'r') as ecg:
        status = pd.DataFrame(
            ecg['meta']['status'][:],
            columns=ecg['meta']['status_keys'][:]
        )
        meta = pd.DataFrame(
            [
                ecg['meta']['alias'][:],
                ecg['meta']['date'][:],
                ecg['meta']['path'][:],
                ecg['meta']['device'][:],
                ecg['meta']['lead_system'][:]
            ],
            index=[
                'alias',
                'date',
                'path',
                'device',
                'lead_system'
            ]
        ).T
        meta.date = pd.to_datetime(meta.date, format="%Y-%m-%d %H:%M:%S")
        return pd.concat([meta, status], axis='columns')
