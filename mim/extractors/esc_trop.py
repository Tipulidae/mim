import numpy as np
import pandas as pd
from scipy.signal import iirnotch, filtfilt, resample

from tqdm import tqdm

from mim.massage.esc_trop import (
    make_mace_table,
    make_index_visits,
    make_ed_features,
    make_lab_features,
    make_double_ecg_features,
    make_ecg_table,
    make_mace30_dataframe,
    make_ami30_dataframe,
    make_mace_chapters_dataframe,
    make_forberg_features,
    _read_esc_trop_csv
)
from mim.extractors.extractor import Data, Container, ECGData, Extractor
from mim.cross_validation import CrossValidationWrapper, ChronologicalSplit
from mim.util.logs import get_logger

log = get_logger("ESC-Trop extractor")


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

        return Data(df.values, columns=list(df))

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
            data = features[self.features['flat_features']].values
            x_dict['flat_features'] = Data(
                data,
                columns=self.features['flat_features'],
            )

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
                        f0.values,
                        columns=list(f0.columns)
                    )
                if 'ecg_1' in self.features['forberg']:
                    x_dict['forberg_ecg_1'] = Data(
                        f1.values,
                        columns=list(f1.columns)
                    )
                if 'diff' in self.features['forberg']:
                    x_dict['forberg_diff'] = Data(
                        diff.values,
                        columns=list(diff.columns)
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
        labels = self.make_labels(index, **self.labels)
        log.debug('Making features')
        feature_dict = self.make_features(index)

        data = Container(
            {
                'x': Container(feature_dict),
                'y': labels,
                'index': Data(index.index, columns=['Alias'])
            },
            index=index.reset_index().index,
            fits_in_memory=self.fits_in_memory
        )
        if self.cv_kwargs is not None and 'test_size' in self.cv_kwargs:
            test_size = self.cv_kwargs['test_size']
        else:
            test_size = 1 / 4
        hold_out_splitter = CrossValidationWrapper(
            ChronologicalSplit(test_size=test_size)
        )
        dev, _ = next(hold_out_splitter.split(data))
        log.debug('Finished extracting esc-trop data')
        return dev


class EscTropECG(Extractor):
    def make_index(self, exclude_new_ecgs=True, exclude_test_patients=True,
                   exclude_index_ecgs=True, **index_kwargs):
        index = make_index_visits(**index_kwargs)
        index_ecgs = make_double_ecg_features(index)
        ecgs = make_ecg_table()

        log.debug(f"{len(ecgs)} usable ECGs")
        if exclude_new_ecgs:
            ecgs = ecgs[ecgs.ecg_date < '2017-02-01']
            log.debug(f"{len(ecgs)} ECGs after excluding 'new' ECGs")
        if exclude_test_patients:
            first_test_patient_index = -(len(index) // 4)
            ecgs = ecgs[~(ecgs.Alias.isin(
                index.iloc[first_test_patient_index:].index))]
            log.debug(f"{len(ecgs)} ECGs after excluding ECGs from test "
                      f"patients")
        if exclude_index_ecgs:
            ecgs = ecgs[
                ~(ecgs.index.isin(index_ecgs.ecg_0)) &
                ~(ecgs.index.isin(index_ecgs.ecg_1))
            ]
            log.debug(f"{len(ecgs)} ECGs after excluding index and 'previous'"
                      f" ECGs")

        ed = _read_esc_trop_csv(
            "ESC_TROP_Vårdkontakt_InkluderadeIndexBesök_2017_2018.csv"
        ).set_index('Alias')

        ecgs = ecgs.join(ed["Kön"], on="Alias").dropna()
        ecgs['male'] = ecgs["Kön"].map({"M": 1, "F": 0})

        return np.array(ecgs.index), ecgs['male']

    def get_data(self) -> Container:
        log.debug('Making index')
        index, male = self.make_index()

        if 'ecg_mode' in self.features:
            mode = self.features['ecg_mode']
        else:
            mode = 'raw'

        x = ECGData(
            '/mnt/air-crypt/air-crypt-esc-trop/axel/ecg.hdf5',
            mode=mode,
            index=index,
            fits_in_memory=self.fits_in_memory
        )
        x = Container({"ecg": x})

        y = Data(male.values, columns=["male"])

        data = Container(
            {'x': x,
             'y': y,
             'index': Data(index, columns=['HDF5_ECG_ID'])},
            index=range(len(y))
        )

        # No need for a hold-out set if we exclude the test-patients.
        log.debug('Finished extracting esc-trop ECGs')
        return data


def preprocess_ecg(ecg_data, processing, scale=5000):
    data = ecg_data.as_numpy()

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
    return Data(data, columns=ecg_data.columns)


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


def calculate_four_last_leads(data):
    data = np.pad(data, pad_width=[[0, 0], [0, 4]])
    i = data[:, 6]
    ii = data[:, 7]
    iii = ii - i
    avr = -(i+ii)/2
    avl = (i-iii)/2
    avf = (ii+iii)/2

    data[:, 8] = iii
    data[:, 9] = avr
    data[:, 10] = avl
    data[:, 11] = avf
    return data
