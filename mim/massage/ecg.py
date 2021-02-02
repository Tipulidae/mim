import glob

import scipy.io as sio
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from mim.massage.carlson_ecg import ECGStatus, ecg_status


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
            return self.ecg_dict['Recording']['Device'][0][0][0]
        else:
            return "Unknown device"

    def init_lead_system(self):
        if self.is_lead_system_ok:
            return self.ecg_dict['Recording']['Lead_system'][0][0][0]
        else:
            return "Unknown lead system"

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
            ECGStatus.MISSING_LEAD_SYSTEM
        })


def create_esc_trop_ecg_hdf5():
    # TODO:
    #  Refactor this, maybe add a metadata report to it, and run once more
    #  just to be safe.
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
            (n, 25),
            dtype=np.bool
        )
        meta.create_dataset(
            "status_keys",
            (25,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )

        data['meta']['status_keys'][()] = [x.name for x in ECGStatus]
        data['meta']['path'][()] = ecg_paths

        for i, path in tqdm(enumerate(ecg_paths), total=n):
            ecg = ECG(path)
            data['raw'][i] = ecg.raw
            data['beat'][i] = ecg.beat
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
