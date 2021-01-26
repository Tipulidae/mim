# -*- coding: utf-8 -*-

from datetime import datetime
import gzip
import json
# from sklearn.metrics import roc_auc_score
#
# import tensorflow as tf


def get_opener(filename):
    if filename.endswith(".gz") or filename.endswith(".gzip"):
        return gzip.open
    else:
        return open


def load_json(filename):
    with get_opener(filename)(filename, "rt", encoding="utf8") as fid:
        data = [json.loads(line) for line in fid]
    return data


def parse_iso8601_datetime(s) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


# def sklearn_auc( y_true, y_pred ) :
#     score = tf.py_function(lambda y_true, y_pred : roc_auc_score(
#     y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
#                         [y_true, y_pred],
#                         'float32',
#                         name='sklearnAUC' )
#     return score
#
# class StreamingScikitLearnAUCMetric(tf.keras.metrics.Metric):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.state = self.add_weight(name="state", initializer="zeros")
#         self.foo = self.add_weight(name="bar", initializer="zeros")
#         self.foo.assign_add(0.1)
#
#     def result(self):
#         return self.state
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         #print("Update state")
#         print(y_true.numpy)
#         self.state.assign_add(self.foo)
#
#     def reset_states(self):
#         super().reset_states()
#         self.foo.assign_add(0.1)
# #        print("reset states")
# #        self.state = 0
