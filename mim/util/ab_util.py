# -*- coding: utf-8 -*-

import gzip
import json


def get_opener(filename):
    if filename.endswith(".gz") or filename.endswith(".gzip"):
        return gzip.open
    else:
        return open


def load_json(filename):
    with get_opener(filename)(filename, "rt", encoding="utf8") as fid:
        data = [json.loads(line) for line in fid]
    return data
