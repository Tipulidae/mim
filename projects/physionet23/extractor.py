from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
from tqdm import tqdm

from massage.physionet2023 import extract_brain_waves_from_patient
from mim.experiments.extractor import Extractor, DataWrapper, Data, Container
from mim.util.logs import get_logger
from mim.cache.decorator import cache
from massage import physionet2023 as pn23


log = get_logger("Physionet 2023 Extractor")


def make_features(index, patient_features=True, eeg_features=None):
    x_dict = {}
    if patient_features:
        log.debug("Making patient features")
        features = pn23.make_patient_features(index)
        x_dict['patient_features'] = Data(
            features.values,
            columns=features.columns,
        )

    if eeg_features is not None:
        x_dict['eeg_features'] = make_eeg_features(index, **eeg_features)

    log.debug("Finished making features")
    return Container(x_dict)


def make_eeg_features(index, lstm=False, brain_waves=None, autoencoder=None):
    eeg_features = []

    if brain_waves is not None:
        log.debug("Making brain-wave features")
        features = make_brainwave_features(index, **brain_waves)
        if lstm:
            features = fix_time_dimension(features, **brain_waves)
        eeg_features.append(features)

    if autoencoder is not None:
        log.debug("Making auto-encoder features")
        features = make_autoencoder_features(index, **autoencoder)
        if lstm:
            features = fix_time_dimension(features, **autoencoder)
        eeg_features.append(features)

    return Data(np.concatenate(eeg_features, axis=-1))


def fix_time_dimension(features, which='all', **kwargs):
    if which == 'all':
        eeg_dimension_len = 72
    elif isinstance(which, list):
        eeg_dimension_len = len(which)
    else:
        eeg_dimension_len = 1

    return features.reshape((features.shape[0], eeg_dimension_len, -1))


def make_autoencoder_features(index, **settings):
    return np.array([])


@cache
def make_brainwave_features(index, **settings):
    workers = cpu_count() - 1
    with Pool(workers) as p:
        features = list(tqdm(
            p.imap(partial(extract_brain_waves_from_patient, **settings),
                   index.index),
            total=len(index),
            desc='Extracting brain-waves'
        ))

    return np.stack(features)


def make_labels(index, outcome=True, cpc=False):
    labels = pn23.make_labels(index)
    columns = []
    if outcome:
        columns.append('outcome')
    if cpc:
        columns.append('cpc')

    return Data(
        labels.loc[:, columns].values,
        columns=columns,
    )


class ICARE(Extractor):
    def get_development_data(self) -> DataWrapper:
        index = pn23.make_index()

        return DataWrapper(
            features=make_features(index, **self.features),
            labels=make_labels(index, **self.labels),
            index=Data(index.index.values, columns=['PatientID']),
            fits_in_memory=True,
        )

    def get_test_data(self) -> DataWrapper:
        # We don't need a hold-out test set in this case, since it is done
        # by the organizers of the physionet challenge.
        raise NotImplementedError
