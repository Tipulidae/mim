import numpy as np
import pandas as pd
from scipy.signal import iirnotch, filtfilt, resample

from tqdm import tqdm
import h5py

from massage.ecg import calculate_four_last_leads
from mim.experiments.extractor import Extractor, DataWrapper, Data, Container
from mim.cross_validation import CrossValidationWrapper, ChronologicalSplit
from mim.util.logs import get_logger
from massage.esc_trop import (
    make_mace_table,
    make_index_visits,
    make_ed_features,
    make_lab_features,
    make_double_ecg_features,
    make_mace30_dataframe,
    make_ami30_dataframe,
    make_mace_chapters_dataframe,
    make_forberg_features, make_johansson_features,
)

log = get_logger("ESC-Trop extractor")
ECG_PATH = '/mnt/air-crypt/air-crypt-esc-trop/axel/ecg.hdf5'
ECG_COLUMNS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'I', 'II']


class EscTrop(Extractor):
    def make_index(self):
        return make_index_visits(**self.index)

    def make_labels(self, index, target='mace30', **kwargs):
        assert target in ['mace30', 'ami30', 'mace_chapters']
        mace_table = make_mace_table(index, **kwargs)

        if target == 'mace30':
            df = make_mace30_dataframe(mace_table)
        elif target == 'ami30':
            df = make_ami30_dataframe(mace_table)
        else:
            df = make_mace_chapters_dataframe(mace_table)

        assert index.index.equals(df.index)

        return Data(df.values, columns=df.columns)

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
                    x_dict[ecg] = Data(
                        self.make_ecg_data(ecg_features[ecg]),
                        columns=ECG_COLUMNS
                    )

        if 'flat_features' in self.features:
            data = features[self.features['flat_features']]
            x_dict['flat_features'] = Data(data.values, columns=data.columns)

        if 'forberg' in self.features:
            f0 = make_forberg_features(ecg_features.ecg_0)
            f1 = make_forberg_features(ecg_features.ecg_1)
            diff = (f1 - f0)

            if 'combine' in self.features['forberg']:
                # This is just a hack to allow the forberg-features to be
                # combined into a single vector, rather than split into
                # separate vectors in the Data dict. This is so that they can
                # be pre-processed together by PCA, rather than separately.
                # Meanwhile, I need to be able to have the features from each
                # ECG in a separate vector for the FFNN-model to work
                # properly, hence the switch here.
                values = []
                columns = []
                if 'ecg_0' in self.features['forberg']:
                    values.append(f0.values)
                    columns += [f"{name}_ecg_0" for name in f0.columns]

                if 'ecg_1' in self.features['forberg']:
                    values.append(f1.values)
                    columns += [f"{name}_ecg_1" for name in f1.columns]

                if 'diff' in self.features['forberg']:
                    values.append(diff.values)
                    columns += [f"{name}_diff" for name in diff.columns]

                x_dict['forberg_features'] = Data(
                    np.concatenate(values, axis=1),
                    columns=columns
                )
            else:
                if 'ecg_0' in self.features['forberg']:
                    x_dict['forberg_ecg_0'] = Data(
                        f0.values, columns=f0.columns)
                if 'ecg_1' in self.features['forberg']:
                    x_dict['forberg_ecg_1'] = Data(
                        f1.values, columns=f1.columns)
                if 'diff' in self.features['forberg']:
                    x_dict['forberg_diff'] = Data(
                        diff.values, columns=diff.columns)

        if 'johansson' in self.features:
            f0 = make_johansson_features(ecg_features.ecg_0)
            f1 = make_johansson_features(ecg_features.ecg_1)
            diff = (f1 - f0)

            if 'combine' in self.features['johansson']:
                values = []
                columns = []
                if 'ecg_0' in self.features['johansson']:
                    values.append(f0.values)
                    columns += [f"{name}_ecg_0" for name in f0.columns]

                if 'ecg_1' in self.features['johansson']:
                    values.append(f1.values)
                    columns += [f"{name}_ecg_1" for name in f1.columns]

                if 'diff' in self.features['johansson']:
                    values.append(diff.values)
                    columns += [f"{name}_diff" for name in diff.columns]

                x_dict['johansson_features'] = Data(
                    np.concatenate(values, axis=1),
                    columns=columns
                )
            else:
                if 'ecg_0' in self.features['johansson']:
                    x_dict['johansson_ecg_0'] = Data(
                        f0.values, columns=f0.columns)
                if 'ecg_1' in self.features['johansson']:
                    x_dict['johansson_ecg_1'] = Data(
                        f1.values, columns=f1.columns)
                if 'diff' in self.features['johansson']:
                    x_dict['johansson_diff'] = Data(
                        diff.values, columns=diff.columns)
        return x_dict

    def make_ecg_data(self, ecgs):
        mode = self.features['ecg_mode']
        with h5py.File(ECG_PATH, 'r') as f:
            data = np.array([f[mode][ecg] for ecg in ecgs])

        return preprocess_ecg(data, self.processing)

    def get_data(self):
        log.debug('Making index')
        index = self.make_index()
        log.debug('Making labels')
        labels = self.make_labels(index, **self.labels)
        log.debug('Making features')
        feature_dict = self.make_features(index)

        data = DataWrapper(
            features=Container(feature_dict),
            labels=labels,
            index=Data(index.index.values, columns=['Alias']),
            groups=index.index.values,
            fits_in_memory=self.fits_in_memory
        )

        if self.cv_kwargs is not None and 'test_size' in self.cv_kwargs:
            test_size = self.cv_kwargs['test_size']
        else:
            test_size = 1 / 4
        hold_out_splitter = CrossValidationWrapper(
            ChronologicalSplit(test_size=test_size)
        )
        development_data, test_data = next(hold_out_splitter.split(data))
        log.debug('Finished extracting esc-trop data')
        return development_data, test_data

    def get_development_data(self) -> DataWrapper:
        development_data, _ = self.get_data()
        return development_data

    def get_test_data(self) -> DataWrapper:
        _, test_data = self.get_data()
        return test_data


def preprocess_ecg(data, processing, scale=5000):
    if processing is not None:
        if 'scale' in processing:
            scale = processing['scale']
        if 'notch-filter' in processing:
            filtered_data = np.zeros_like(data)
            for ecg in tqdm(range(len(data)), desc='applying notch-filter'):
                filtered_data[ecg] = notch_ecg(data[ecg])
                filtered_data[ecg] -= np.median(filtered_data[ecg], axis=0)

            data = filtered_data
        if 'clip_outliers' in processing:
            data = np.clip(data, -0.0004, 0.0004)

        if 'ribeiro' in processing:
            shape = (len(data), 4096, 12)
            fixed_data = np.zeros(shape)

            for ecg in tqdm(range(len(data)), desc='resampling ecgs'):
                fixed_data[ecg] = resample_pad_and_fix_leads(data[ecg])

            data = fixed_data

    # I will scale regardless, as not doing so will lead to problems.
    data *= scale
    return data


def notch_ecg(data):
    filtered_data = np.zeros_like(data)
    for lead in range(8):
        b, a = iirnotch(0.05, Q=0.005, fs=1000)
        filtered_data[:, lead] = filtfilt(b, a, data[:, lead])

    return filtered_data


def resample_pad_and_fix_leads(data):
    # Resample to 400Hz
    # pad to 4096 (adding zeros on both side)
    # add the 4 missing leads
    # Reorder the leads so that they come in the order expected by
    # Ribeiro et al.
    # Maybe re-scale data to be in V instead of mV?
    data = resample(data, 4000)
    data = np.pad(data, pad_width=[[48, 48], [0, 0]])
    data = calculate_four_last_leads(data)
    data = data[:, [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]]
    return data


def resample_to_400hz(data):
    resample(data, 4000)
