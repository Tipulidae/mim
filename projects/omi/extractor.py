import os
import string

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from mim.cross_validation import CrossValidationWrapper
from mim.experiments.extractor import (
    Extractor,
    Augmentor,
    DataWrapper,
    Container,
    Data
)
from mim.util.logs import get_logger
from mim.config import PATH_TO_DATA
from massage import esc_trop

log = get_logger("OMI-Extractor")
ECG_COLUMNS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'I', 'II']


def make_ecg_data(ecg_ids):
    mode = 'raw'
    with h5py.File(esc_trop.ECG_PATH, 'r') as f:
        data = np.array([
            f[mode][ecg] for ecg in
            tqdm(ecg_ids, desc=f'Extracting {mode} ECG data')
        ])

    data *= 5000
    return data


def make_angiographies(admission_dates):
    # Admission dates should be a dataframe with 1 column called
    # admission_date and index Alias
    log.debug("Loading sectra files")
    sectra = esc_trop.load_sectra()

    log.debug("Finding angiographies")
    angiography_codes = ['37300', '39648', '39600']
    sectra_angio = (
        sectra
        .loc[sectra.sectra_code.isin(angiography_codes), :]
        .join(admission_dates, how='inner', on='Alias')
    )
    dt = (sectra_angio.sectra_date -
          sectra_angio.admission_date).dt.total_seconds() / (3600 * 24)
    sectra_angio = (
        sectra_angio[(dt >= -1) & (dt <= 7)]
        .sort_values(by=['Alias', 'sectra_date'])
        .groupby('Alias')
        .agg({'sectra_date': 'first', 'answer': '\n\n'.join})
    )
    sectra_angio['has_angio'] = True
    return (
        pd.DataFrame(index=admission_dates.index)
        .join(sectra_angio)
        .fillna({'answer': 'N/A', 'has_angio': False})
    )


def make_annotation_documents():
    document_path = os.path.join(PATH_TO_DATA, 'sectra_documents')
    log.debug("Loading esc-trop index")
    index = esc_trop.make_index_visits(
        exclude_stemi=False,
        exclude_missing_tnt=False,
        exclude_missing_ecg=True,
        exclude_missing_old_ecg=False,
        exclude_missing_chest_pain=True,
        exclude_non_swedish_patients=True,
    )
    log.debug("Creating OMI table")
    omi = esc_trop.make_omi_table(index)
    omi['proxy_omi'] = esc_trop.make_omi_label(
        omi, stenosis_limit=90, tnt_limit=750)
    log.debug("Loading sectra files")
    sectra = esc_trop.load_sectra()

    log.debug("Finding angiographies")
    angiography_codes = ['37300', '39648', '39600']
    sectra_angio = (
        sectra
        .loc[sectra.sectra_code.isin(angiography_codes), :]
        .join(omi.admission_date, how='inner', on='Alias')
    )
    dt = (sectra_angio.sectra_date -
          sectra_angio.admission_date).dt.total_seconds() / (3600 * 24)
    sectra_angio = (
        sectra_angio[(dt >= -1) & (dt <= 7)]
        .sort_values(by=['Alias', 'sectra_date'])
        .groupby('Alias')
        .agg({'sectra_date': 'first', 'answer': '\n\n'.join})
    )
    sectra_angio['sectra'] = True
    omi = omi.join(sectra_angio).fillna({'answer': 'N/A', 'sectra': False})

    log.debug("Preparing annotation documents")
    cols = [
        'admission_date', 'occlusion_less_than_3_months_old',
        'stenosis_100%', 'stenosis_90-99%', 'acs_indication', 'stemi',
        'i21_rikshia', 'i21_sos', 'tnt_rikshia', 'tnt_melior',
        'tnt_melior_date', 'prior_cabg', 'proxy_omi', 'answer'
    ]
    documents = omi.loc[omi.sectra & omi.i21, cols].reset_index(drop=True)
    documents = documents.rename(columns={'stemi': 'stemi_rikshia'})

    def num_to_label(i):
        if not 0 <= i <= 1500:
            raise ValueError('number must be between 0 and 1500')

        return f"{string.ascii_uppercase[i//100]}{i%100:02d}"

    for i, row in tqdm(documents.iterrows(),
                       desc='Writing annotation documents'):
        omi_stats = [f"{k:<32}{str(v):>19}" for k, v in row.head(-1).items()]
        text = "<pre>" + "\n".join(omi_stats) + "</pre>"
        text += '<br>' + row.answer.replace('\n', '<br>')

        with open(os.path.join(document_path, f"{num_to_label(i)}.html"), 'w',
                  encoding="utf-8") as f:
            f.write(text)


class OMIExtractor(Extractor):
    def get_data(self):
        log.debug('Making index')
        index = esc_trop.make_index_visits(
            exclude_stemi=False,
            exclude_missing_tnt=False,
            exclude_missing_ecg=True,
            exclude_missing_old_ecg=False,
            exclude_missing_chest_pain=True,
            exclude_non_swedish_patients=True,
        )

        log.debug('Making labels')
        omi_table = esc_trop.make_omi_table(index)
        # y = make_omi_label(omi_table, stenosis_limit=90, tnt_limit=750)
        y = esc_trop.make_omi_label(omi_table, **self.labels)

        log.debug('Making features')
        ecg_features = esc_trop.make_double_ecg_features(index)
        x_dict = {
            'ecg_0': Data(
                make_ecg_data(ecg_features.ecg_0),
                columns=ECG_COLUMNS
            )
        }

        if not index.index.equals(y.index):
            raise ValueError('Index and targets are different!')

        data = DataWrapper(
            features=Container(x_dict),
            labels=Data(y.values, columns=['OMI']),
            index=Data(index.index.values, columns=['Alias']),
            groups=index.index.values,
            fits_in_memory=self.fits_in_memory
        )

        if self.cv_kwargs is not None and 'test_size' in self.cv_kwargs:
            test_size = self.cv_kwargs['test_size']
        else:
            test_size = 1 / 4
        hold_out_splitter = CrossValidationWrapper(
            StratifiedShuffleSplit(test_size=test_size)
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


class UseAdditionalECGs(Augmentor):
    def augment_training_data(self, data: DataWrapper) -> DataWrapper:
        index = esc_trop.make_index_visits(
            exclude_stemi=False,
            exclude_missing_tnt=False,
            exclude_missing_ecg=False,
            exclude_missing_old_ecg=False,
            exclude_missing_chest_pain=True,
            exclude_non_swedish_patients=True,
        )

        old_labels = data.y
        # Because we're only using some of the data!
        index = index.loc[old_labels.index, :]

        # Kinda convoluted, there is probably a better way to do this...
        all_ecgs = esc_trop.make_ed_ecgs(index)[[
            'Alias', 'ecg']].set_index('Alias')
        old_index = all_ecgs.groupby('Alias').head(1).loc[old_labels.index, :]
        new_index = all_ecgs.groupby('Alias').tail(-1)
        full_index = pd.concat([old_index, new_index], axis=0)

        log.info(f"Found {len(new_index)} additional ECGs to use for "
                 f"augmentation")

        old_labels = data.y
        new_labels = new_index.join(old_labels).OMI
        all_labels = np.concatenate(
            [old_labels.OMI.values, new_labels.values], axis=0)

        new_ecg_data = make_ecg_data(new_index.ecg)

        x_dict = {
            'ecg_0': Data(
                np.concatenate([
                    data.data['x']['ecg_0'].as_numpy(),
                    new_ecg_data
                ]),
                columns=ECG_COLUMNS
            )
        }

        return DataWrapper(
            features=Container(x_dict),
            labels=Data(
                all_labels,
                columns=['OMI']
            ),
            index=Data(full_index.index.values, columns=['Alias']),
            groups=full_index.index.values,
            fits_in_memory=data.data.fits_in_memory
        )
