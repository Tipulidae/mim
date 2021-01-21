# -*- coding: utf-8 -*-

import numpy as np

from mim.extractors.extractor import Data, Container
import mim.util.ab_util
from mim.util.ab_util import parse_iso8601_datetime

CATEGORICAL_FEATURES = ["gender"]
NUMERICAL_FEATURES = ["age"] + ["bl-" + sample
                                for sample in
                                ["Glukos", "Krea", "TnT", "Hb"]
                                ]


class JSONDataPoint:
    def __init__(self, m, ecg=None):
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
            self.ecg = ecg
        else:
            self.ecg_timestamp = None
            self.ecg = None

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


class ABJSONExtractor:
    def __init__(self, specification):
        self.specification = specification

    def get_data(self):
        json_data = _parse_json(self.specification)
        x_container_dict = _extract_x(json_data, self.specification)
        y = _get_labels(json_data, self.specification)
        data = Container(
            {
                "x": Container(x_container_dict),
                "y": Data(y)
            }
        )
        return data


def _extract_x(json_data, specification):
    r = {}
    print(specification["features"])
    numerical_keys = sorted(list(set(
        specification["features"]).intersection(NUMERICAL_FEATURES)))
    categorical_keys = sorted(list(set(
        specification["features"]).intersection(CATEGORICAL_FEATURES)))
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
    return r


def _get_categorical(dp: JSONDataPoint, categorical_keys):
    r = []
    for k in categorical_keys:
        if k == "gender":
            r += [dp.is_male, dp.is_female]
        else:
            raise ValueError("Unknown key: " + k)
    return r


def _get_labels(json_data, specification):
    return np.array([1 if d[specification["labels"]["target"]] == "T" else 0
                     for d in json_data])


def _parse_json(specification):
    file = specification["index"]["json_train"]
    data = mim.util.ab_util.load_json(file)
    return data
