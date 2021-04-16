import numpy as np
from scipy.signal import iirnotch, filtfilt

from tqdm import tqdm

from mim.massage.esc_trop import make_double_ecg_table
from mim.extractors.extractor import Data, Container, ECGData, Extractor
from mim.cross_validation import CrossValidationWrapper, ChronologicalSplit
from mim.util.logs import get_logger

log = get_logger("ESC-Trop extractor")


class EscTrop(Extractor):
    def get_data(self) -> Container:
        log.debug("Preparing ED and TnT data...")
        ed = make_double_ecg_table()

        ecg_path = '/mnt/air-crypt/air-crypt-esc-trop/axel/ecg.hdf5'
        mode = self.features['ecg_mode']

        x_dict = {}
        if 'index' in self.features['ecgs']:
            ecgs = ECGData(
                ecg_path,
                mode=mode,
                index=ed.ecg_id.astype(int).values
            )
            x_dict['ecg_0'] = preprocess_ecg(ecgs, self.processing)
        if 'old' in self.features['ecgs']:
            ecgs = ECGData(
                ecg_path,
                mode=mode,
                index=ed.old_ecg_id.astype(int).values
            )
            x_dict['ecg_1'] = preprocess_ecg(ecgs, self.processing)
        if 'features' in self.features:
            x_dict['features'] = Data(ed[self.features['features']].values)

        data = Container(
            {
                'x': Container(x_dict),
                'y': Data(ed.mace_30_days.astype(int).values)
            },
            index=ed.index,
            fits_in_memory=self.fits_in_memory
        )

        hold_out_splitter = CrossValidationWrapper(
            ChronologicalSplit(test_size=1/4)
        )
        dev, _ = next(hold_out_splitter.split(data))
        return dev


def preprocess_ecg(ecg_data, processing):
    data = ecg_data.as_numpy

    if processing is not None:
        if 'notch-filter' in processing:
            filtered_data = np.zeros_like(data)
            for ecg in tqdm(range(len(data)), desc='applying notch-filter'):
                filtered_data[ecg] = notch_ecg(data[ecg])
                filtered_data[ecg] -= np.median(filtered_data[ecg], axis=0)

            data = filtered_data
        if 'clip_outliers' in processing:
            data = np.clip(data, -0.0004, 0.0004)

    # I will scale regardless, as not doing so will lead to problems.
    data *= 5000
    return Data(data)


def notch_ecg(data):
    filtered_data = np.zeros_like(data)
    for lead in range(8):
        b, a = iirnotch(0.05, Q=0.005, fs=1000)
        filtered_data[:, lead] = filtfilt(b, a, data[:, lead])

    return filtered_data
