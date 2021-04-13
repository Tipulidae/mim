# -*- coding: utf-8 -*-
from typing import List

import numpy as np
import tensorflow as tf

from mim.extractors.extractor import Data, Container, Extractor
from mim.massage.ecg import ECG
import mim.util.ab_util
from mim.util.ab_util import parse_iso8601_datetime

CATEGORICAL_FEATURES = ["gender"]
NUMERICAL_FEATURES = ["age"] + [f"bl-{sample}" for sample in
                                ["Glukos", "Krea", "TnT", "Hb"]
                                ]
ECG_FEATURES = ["ecg_raw_12", "ecg_raw_8", "ecg_beat_12", "ecg_beat_8"]


class JSONDataPoint:
    def __init__(self, m):
        (self.is_female, self.is_male) = (1, 0) if m["gender"] == "Female" \
            else (0, 1)
        (tnt1, tnt2, tnt_time_delta_seconds) = self.extract_tnts(m)
        self.m = m
        self.guid = m["id"] if "id" in m else None
        self.origin = m["data_origin"] if "data_origin" in m else None
        self.dpid = m["dpid"] if "dpid" in m else "undef"
        self.tnt1_timestamp = parse_iso8601_datetime(m["tnts"][0][0]) \
            if len(m["tnts"]) > 0 else None
        self.index_datetime = parse_iso8601_datetime(m["index_datetime"]) \
            if "index_datetime" in m else self.tnt1_timestamp
        self.reals_absolute = {
            "age": float(m["age"]),
            "tnt1": float(tnt1),
            "tnt2": float(tnt2),
            "tnt_time_delta_seconds":
                tnt_time_delta_seconds,
            "charlson":
                float(m["Charlson-Score"]) if "Charlson-Score" in m else None
        }
        for (key, value) in m["blood_samples"].items():
            if value:
                self.reals_absolute["bl-" + key] = value
        if "ecg" in m and m["ecg"] is not None:
            self.ecg_timestamp = parse_iso8601_datetime(m["ecg"][0])
            self.ecg_path = m["ecg"][1]
        else:
            self.ecg_timestamp = None

    @staticmethod
    def extract_tnts(m):
        tnts = m["tnts"]
        if len(tnts) == 0:
            return np.nan, np.nan, np.nan
        (date1, value1) = tnts[0]
        if len(tnts) < 2:
            return value1, np.nan, np.nan
        (date2, value2) = tnts[1]
        time_delta_seconds = (parse_iso8601_datetime(date2) -
                              parse_iso8601_datetime(date1)).total_seconds()
        return value1, value2, time_delta_seconds


class ABJSONExtractor(Extractor):
    def get_data(self) -> Container:
        train_json = []
        for train_file in self.index['train_files']:
            train_json.extend(_parse_json(train_file))

        val_json = []
        for val_file in self.index['val_files']:
            val_json.extend(_parse_json(val_file))

        dev_json = train_json + val_json

        x = _extract_x(dev_json, self.features)

        data = Container(
            {
                'x': Container(x, index=range(len(dev_json))),
                'y': Data(_get_labels(dev_json, self.labels))
            },
            predefined_splits=_define_split(len(train_json), len(val_json))
        )

        return data


def _define_split(train_len, val_len):
    """
    The entry ``split[i]`` represents the index of the test set that
    sample ``i`` belongs to. It is possible to exclude sample ``i`` from
    any test set (i.e. include sample ``i`` in every training set) by
    setting ``split[i]`` equal to -1.

    This function defines a split by setting the first train_len elements to
    -1, thus always including them in the training set, and then setting the
    remaining val_len elements to 0, using them in the test set for the first
    (and only) split.
    """
    split = [-1] * train_len + [0] * val_len
    return split


class LazyECGsFromFiles(Data):

    def __init__(self, key, filenames: List, caching=True):
        self.filenames = filenames
        self.cache = {} if caching else None
        _, s, n = key.split("_", 2)
        self.last_lead = 8 if n == "8" else 12
        super().__init__(data=filenames, index=list(range(len(filenames))),
                         dtype=tf.float32, fits_in_memory=True)
        if s == "raw":
            self.raw_signal = True
            self._shape = [10000, self.last_lead]
        else:
            self.raw_signal = False
            self._shape = [1200, self.last_lead]

    def _load(self, filename):
        ecg = ECG(filename)
        if self.raw_signal:
            matrix = ecg.ecg_dict["Data"]["ECG"][0][0][:10000, :self.last_lead]
        else:
            matrix = ecg.ecg_dict["Data"]["Beat"][0][0][:1200, :self.last_lead]
        return matrix

    def _cached(self, item):
        if item in self.cache:
            return self.cache[item]
        else:
            ecg = self._load(self.filenames[item])
            self.cache[item] = ecg
            return ecg

    def __getitem__(self, item):
        if self.cache is not None:
            return self._cached(self._index[item])
        else:
            return self._load(self.filenames[self._index[item]])

    @property
    def cache_size(self):
        if self.cache is not None:
            return len(self.cache)
        else:
            return 0

    def clear_cache(self):
        if self.cache is not None:
            self.cache.clear()


def _extract_x(json_data, features):
    r = {}
    print(features)
    numerical_keys = sorted(list(set(
        features).intersection(NUMERICAL_FEATURES)))
    categorical_keys = sorted(list(set(
        features).intersection(CATEGORICAL_FEATURES)))
    ecg_keys = sorted(list(set(features).intersection(ECG_FEATURES)))
    dps = list(map(JSONDataPoint, json_data))
    if numerical_keys:
        print(numerical_keys)
        r["numeric"] = Data(
            np.array([[dp.reals_absolute[k] for k in numerical_keys]
                      for dp in dps])
        )
    if categorical_keys:
        print(categorical_keys)
        r["categorical"] = Data(
            np.array([_get_categorical(dp, categorical_keys) for dp in dps])
        )
    if ecg_keys:
        assert len(ecg_keys) == 1
        files = [dp.ecg_path for dp in dps]
        r["ecg"] = LazyECGsFromFiles(ecg_keys[0], files)

    return r


def _get_categorical(dp: JSONDataPoint, categorical_keys):
    r = []
    for k in categorical_keys:
        if k == "gender":
            r += [dp.is_male, dp.is_female]
        else:
            raise ValueError("Unknown key: " + k)
    return r


def _get_labels(json_data, labels):
    return np.array([1 if d[labels["target"]] == "T" else 0
                     for d in json_data])


def _parse_json(json_file):
    data = mim.util.ab_util.load_json(json_file)
    return data
