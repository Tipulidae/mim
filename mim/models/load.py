import os

import pandas as pd
from tensorflow import keras

from mim.config import PATH_TO_TEST_RESULTS
from mim.util.metadata import Validator


def load_keras_model(base_path, split_number, **kwargs):
    path = os.path.join(
        base_path,
        f"split_{split_number}",
        "last.ckpt"
    )
    return keras.models.load_model(filepath=path)


def load_model_from_experiment_result(
        xp_name, commit=None, which='best', split_number=0, trainable=False,
        final_layer_index=-1):
    xp_base_path = os.path.join(
        PATH_TO_TEST_RESULTS,
        xp_name
    )
    xp_results_path = os.path.join(
        xp_base_path,
        'results.pickle'
    )
    xp_model_path = os.path.join(
        xp_base_path,
        f'split_{split_number}',
        f'{which}.ckpt'
    )
    metadata = pd.read_pickle(xp_results_path)['metadata']
    expected_metadata = {
        'has_uncommitted_changes': False,
        'current_commit': commit
    }
    v = Validator(
        allow_different_commits=False,
        allow_uncommitted=True
    )
    v.validate_consistency([metadata, expected_metadata])

    model = keras.models.load_model(filepath=xp_model_path)
    model.trainable = trainable

    model = keras.Model(model.input, model.layers[final_layer_index].output)
    return model
