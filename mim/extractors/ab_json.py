# -*- coding: utf-8 -*-

import numpy as np

from mim.extractors.extractor import Data, Container
import mim.util.ab_util

CATEGORICAL_FEATURES = {
    "gender": lambda j: [0, 1] if j["gender"] == "M" else [1, 0]
}

NUMERICAL_FEATURES = {
    "age": lambda j: j["age"]
}
for bs in ["Glukos", "Krea", "TnT"]:
    NUMERICAL_FEATURES["bl-"+bs] = lambda j: j["blood_samples"][bs]
bs = "Foo"
# NUMERICAL_FEATURES["bl-Krea"] = lambda j: j["blood_samples"]["Krea"]
# NUMERICAL_FEATURES["bl-Glukos"] = lambda j: j["blood_samples"]["Glukos"]
# NUMERICAL_FEATURES["bl-TnT"] = lambda j: j["blood_samples"]["TnT"]


def foo(k, j):
    return j[k]


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
    numerical_keys = sorted(list(set(
        specification["features"]).intersection(NUMERICAL_FEATURES)))
    categorical_keys = sorted(list(set(
        specification["features"]).intersection(CATEGORICAL_FEATURES)))
    if numerical_keys:
        print(numerical_keys)
        r["numeric"] = np.array([[NUMERICAL_FEATURES[k](j)
                                  for k in numerical_keys]
                                for j in json_data])
    if categorical_keys:
        def flatten(t):
            return [item for sublist in t for item in sublist]
        r["categorical"] = np.array([flatten([CATEGORICAL_FEATURES[k](j)
                                              for k in categorical_keys])
                                     for j in json_data])
    return r


def _get_labels(json_data, specification):
    return np.array([1 if d[specification["target_label"]] == "T" else 0
                    for d in json_data])


def _parse_json(specification):
    file = specification["json_source"]
    data = mim.util.ab_util.load_json(file)
    return data
