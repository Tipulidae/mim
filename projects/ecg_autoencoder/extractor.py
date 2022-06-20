import numpy as np

from mim.extractors.extractor import (
    Extractor,
    DataWrapper,
    Data,
    Container,
    ECGData
)
from mim.util.logs import get_logger
from massage.esc_trop import (
    make_index_visits,
    make_double_ecg_features,
    make_ecg_table,
    read_csv
)

log = get_logger("ECG Autoencoder Extractor")
ECG_PATH = '/mnt/air-crypt/air-crypt-esc-trop/axel/ecg.hdf5'


# TODO: This requires a bit more thought with the new DataWrapper class.
# But I don't need it right now, so I postpone fixing it.
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
            log.debug(f"{len(ecgs)} ECGs after excluding index "
                      f"and 'previous' ECGs")

        ed = read_csv(
            "ESC_TROP_Vårdkontakt_InkluderadeIndexBesök_2017_2018.csv"
        ).set_index('Alias')

        ecgs = ecgs.join(ed["Kön"], on="Alias").dropna()
        ecgs['male'] = ecgs["Kön"].map({"M": 1, "F": 0})

        return np.array(ecgs.index), ecgs['male']

    def get_data(self) -> DataWrapper:
        log.debug('Making index')
        index, male = self.make_index()

        if 'ecg_mode' in self.features:
            mode = self.features['ecg_mode']
        else:
            mode = 'raw'

        x = ECGData(
            ECG_PATH,
            mode=mode,
            index=index,
            fits_in_memory=self.fits_in_memory
        )
        x = Container({"ecg": x})

        y = Data(male.values, columns=["male"])

        data = Container(
            {'x': x,
             'y': y},
            index=range(len(y))
        )

        # No need for a hold-out set if we exclude the test-patients.
        log.debug('Finished extracting esc-trop ECGs')
        return data
