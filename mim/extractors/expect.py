import json

import pandas as pd
from tensorflow import float64
from sklearn.preprocessing import OrdinalEncoder

from mim.extractors.extractor import Data, Container

CHARLSON_FEATURES = [
    "Charlson-AcuteMyocardialInfarction",
    "Charlson-CongestiveHeartFailure",
    "Charlson-PeripheralVascularDisease",
    "Charlson-CerebralVascularAccident",
    "Charlson-Dementia",
    "Charlson-PulmonaryDisease",
    "Charlson-ConnectiveTissueDisorder",
    "Charlson-PepticUlcer",
    "Charlson-LiverDisease",
    "Charlson-Diabetes",
    "Charlson-DiabetesComplications",
    "Charlson-Parapelgia",
    "Charlson-RenalDisease",
    "Charlson-Cancer",
    "Charlson-MetstaticCancer",
    "Charlson-SevereLiverDisease",
    "Charlson-HIV",
    "Charlson-Score"
]
PREVIOUS_CONDITIONS = [
    "prev5y-AMI",
    "prev5y-COPD",
    "prev5y-Diabetes",
    "prev5y-Heartfail",
    "prev5y-Hypertens",
    "prev5y-Renal",
    "prev5y-PAD",
    "prev5y-UA",
    "prev5y-CABG",
    "prev5y-PCI",
]
ORDINAL_FEATURES = CHARLSON_FEATURES[:-1] + PREVIOUS_CONDITIONS + ['gender']


class Expect:
    def __init__(self, specification):
        self.specification = specification

    def get_data(self):
        data = self._parse_json()
        x = self._extract_features(data)
        y = self._extract_labels(data)

        index = x.index
        data = Container(
            {
                'y': Data(y.values, index=index),
                'x': Data(x.values, index=index, dtype=float64)
            },
            index=index
        )

        return data

    def _parse_json(self):
        path = '/home/sapfo/andersb/ekg_share/json_data/12tnt/hbg+lund-split/'
        file = path + self.specification['index']['source']
        with open(file) as fp:
            data = [json.loads(line) for line in fp.readlines()]

        return data

    def _extract_features(self, data):
        spec = self.specification['features']
        features = []
        if 'troponin' in spec:
            tnts = pd.DataFrame.from_records(data, columns=['tnts'])
            troponin_features = extract_tnt_features(tnts.tnts)
            features.append(troponin_features)
        if 'age' in spec:
            features.append(
                pd.DataFrame.from_records(data, columns=['age'])
            )
        if 'gender' in spec:
            features.append(
                pd.DataFrame.from_records(data, columns=['gender'])
            )
        if 'charlson' in spec:
            features.append(
                pd.DataFrame.from_records(data, columns=CHARLSON_FEATURES)
            )
        if 'previous_conditions' in spec:
            features.append(
                pd.DataFrame.from_records(data, columns=PREVIOUS_CONDITIONS)
            )

        df = pd.concat(features, axis=1)
        df = df.fillna(0)
        used_ordinals = list(set(ORDINAL_FEATURES) & set(df.columns))
        df[used_ordinals] = OrdinalEncoder().fit_transform(df[used_ordinals])
        return df

    def _extract_labels(self, data):
        target_name = self.specification['labels']['target']
        labels = pd.DataFrame.from_records(
            data, columns=[target_name])
        labels = labels.rename(columns={target_name: 'y'})
        return encode_ordinal(labels)


def encode_ordinal(df):
    return pd.DataFrame(
        OrdinalEncoder().fit_transform(df),
        columns=df.columns
    )


def extract_tnt_features(tnts):
    tnts = pd.DataFrame.from_records(tnts, columns=['tnt1', 'tnt2'])
    tnt1 = pd.DataFrame.from_records(tnts.tnt1, columns=['t', 'tnt'])
    tnt2 = pd.DataFrame.from_records(tnts.tnt2, columns=['t', 'tnt'])
    dt = (pd.to_datetime(tnt2.t) - pd.to_datetime(
        tnt1.t)).dt.total_seconds() // 60
    return pd.DataFrame(
        zip(tnt1.tnt, tnt2.tnt, (tnt2.tnt - tnt1.tnt) / dt),
        columns=['tnt', 'tnt_repeat', 'tnt_diff']
    )
