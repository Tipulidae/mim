import json

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class TroponinExtractor:
    def __init__(self, specification):
        self.specification = specification

    def get_data(self):
        data = self._parse_json()
        X = self._extract_features(data)
        y = self._extract_labels(data)
        return X, y

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
            features.append(troponin_features.loc[:, spec['troponin']])
        if 'numerical' in spec:
            features.append(
                pd.DataFrame.from_records(data, columns=spec['numerical'])
            )
        if 'ordinal' in spec:
            ordinals = pd.DataFrame.from_records(data, columns=spec['ordinal'])
            features.append(encode_ordinal(ordinals))

        return pd.concat(features, axis=1)

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
