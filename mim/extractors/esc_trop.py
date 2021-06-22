import numpy as np
import pandas as pd
from scipy.signal import iirnotch, filtfilt

from tqdm import tqdm

from mim.massage.esc_trop import (
    make_mace_table,
    make_index_visits,
    make_ed_features,
    make_lab_features,
    make_double_ecg_features
)
from mim.extractors.extractor import Data, Container, ECGData, Extractor
from mim.cross_validation import CrossValidationWrapper, ChronologicalSplit
from mim.util.logs import get_logger

log = get_logger("ESC-Trop extractor")


class EscTrop(Extractor):
    def make_index(self):
        index = make_index_visits(
            exclude_stemi=True,
            exclude_missing_tnt=True,
            exclude_missing_ecg=True,
            exclude_missing_old_ecg=True
        )
        return index

    def make_labels(self, index):
        mace = make_mace_table(
            index,
            include_interventions=True,
            include_deaths=True,
            icd_definition='new',
            use_melior=False,
            use_sos=True,
            use_hia=True
        )

        assert index.index.equals(mace.index)

        return mace.mace30.astype(int).values

    def make_features(self, index):
        ed_features = make_ed_features(index)
        lab_features = make_lab_features(index)
        ecg_features = make_double_ecg_features(index)

        assert index.index.equals(ed_features.index)
        assert index.index.equals(lab_features.index)
        assert index.index.equals(ecg_features.index)

        features = pd.concat([
            ed_features,
            lab_features,
            ecg_features
        ], axis=1)

        x_dict = {}
        if 'ecgs' in self.features:
            for ecg in [f'ecg_{x}' for x in range(2)]:
                if ecg in self.features['ecgs']:
                    x_dict[ecg] = self.make_ecg_data(ecg_features[ecg])

        if 'flat_features' in self.features:
            x_dict['flat_features'] = Data(
                features[self.features['flat_features']].values,
                columns=self.features['flat_features']
            )

        return x_dict

    def make_ecg_data(self, ecg):
        return preprocess_ecg(
            ECGData(
                '/mnt/air-crypt/air-crypt-esc-trop/axel/ecg.hdf5',
                mode=self.features['ecg_mode'],
                index=ecg,
            ),
            self.processing
        )

    def get_data(self) -> Container:
        log.debug('Making index')
        index = self.make_index()
        log.debug('Making labels')
        labels = self.make_labels(index)
        log.debug('Making features')
        feature_dict = self.make_features(index)

        data = Container(
            {
                'x': Container(feature_dict),
                'y': Data(labels, columns=['mace30']),
                'index': Data(index.index, columns=['Alias'])
            },
            index=index.reset_index().index,
            fits_in_memory=self.fits_in_memory
        )

        hold_out_splitter = CrossValidationWrapper(
            ChronologicalSplit(test_size=1/4)
        )
        dev, _ = next(hold_out_splitter.split(data))
        log.debug('Finished extracting esc-trop data')
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
    return Data(data, columns=ecg_data.columns)


def notch_ecg(data):
    filtered_data = np.zeros_like(data)
    for lead in range(8):
        b, a = iirnotch(0.05, Q=0.005, fs=1000)
        filtered_data[:, lead] = filtfilt(b, a, data[:, lead])

    return filtered_data
