import os

import pandas as pd
from tensorflow import keras

from mim.experiments.extractor import Data, Container
from mim.config import PATH_TO_TEST_RESULTS, PATH_TO_DATA
from mim.util.metadata import Validator
from mim.util.logs import get_logger

log = get_logger('model-loader')


def load_keras_model(base_path, split_number, **kwargs):
    path = os.path.join(
        base_path,
        f"split_{split_number}",
        "last.ckpt"
    )
    return keras.models.load_model(filepath=path)


def load_model_from_experiment_result(
        xp_project, xp_base, xp_name, commit=None, epoch=None, split_number=0,
        trainable=False, final_layer_index=-1, input_key=None, suffix=None,
        **kwargs):
    xp_base_path = os.path.join(
        PATH_TO_TEST_RESULTS,
        xp_project,
        xp_base,
        xp_name
    )
    xp_results_path = os.path.join(
        xp_base_path,
        'train_val_results.pickle'
    )
    xp_model_path = os.path.join(
        xp_base_path,
        f'split_{split_number}',
        'checkpoints',
        f'epoch_{epoch:03d}.keras'
    )
    xp_results = pd.read_pickle(xp_results_path)
    metadata = xp_results.metadata
    expected_metadata = {
        'has_uncommitted_changes': False,
        'current_commit': commit
    }
    v = Validator(
        allow_different_commits=False,
        allow_uncommitted=False
    )
    v.validate_consistency([metadata, expected_metadata])
    log.debug(f'Model path: {xp_model_path}')

    model = keras.models.load_model(filepath=xp_model_path, compile=False)
    model.trainable = trainable

    if suffix is not None:
        for layer in model.layers:
            layer._name += suffix

    if input_key is None:
        inp = model.input
    else:
        inp = model.input[input_key]

    return inp, model.layers[final_layer_index].output


def load_ribeiro_model(freeze_resnet=False, suffix=None):
    """Loads the model pre-trained by Ribeiro et al.
    Ribeiro, A.H., Ribeiro, M.H., Paixão, G.M.M. et al. Automatic
    diagnosis of the 12-lead ECG using a deep neural network.
    Nat Commun 11, 1760 (2020). https://doi.org/10.1038/s41467-020-15432-4
    """
    resnet = keras.models.load_model(
        filepath=os.path.join(PATH_TO_DATA, 'ribeiro_resnet', 'model.hdf5')
    )
    resnet.trainable = not freeze_resnet
    if suffix is not None:
        for layer in resnet.layers:
            layer._name += suffix

    return resnet.input, resnet.layers[-2].output


def pre_process_using_ribeiro(**kwargs):
    resnet_input, resnet_output = load_ribeiro_model()
    model = keras.Model({'ecg_0': resnet_input}, resnet_output)
    return _pre_process_with_model(model)


def pre_process_using_xp(**load_model_kwargs):
    model = load_model_from_experiment_result(**load_model_kwargs)
    return _pre_process_with_model(model)


def _pre_process_with_model(model):
    def pre_process(train, val):
        log.debug('Processing ECGs using pre-trained model')
        processed_train = _process_ecg(model, train)
        processed_val = _process_ecg(model, val)
        log.debug('Finished processing ECGs')
        return processed_train, processed_val

    return pre_process


def _process_ecg(model, data):
    """
    The point of this function is to take all the ecg-data from the input
    dataset and pass it through the given model. The output is our processed
    data, and we return a new dataset that is the same as the input, except
    all the ecg-data is replaced with the processed data instead.

    The reason we do this is to speed up training when using pre-trained
    models, so that the pre-trained model can process the data a single time,
    rather than once per epoch. This should significantly reduce the
    processing and (perhaps more importantly) memory footprint of the final
    model, providing a significant speedup when training such models.

    :param model: a keras model that takes a single ECG as input, and returns
    some tensor (perhaps a vector) as output.
    :param data: a Container object of the usual form, where we expect some
    ecg-records. Should have keys 'x', 'y' and 'index', where 'x' can be an
    appropriately nested container, but containing at least one ECG record
    (otherwise there is no point to this function).
    :return: New Data (Container) object with a similar layout as the input
    data, but where all the ecg fields have been processed using the input
    model.
    """
    # TODO:
    # Maybe add a "data.keys" or similar and verify that the format is
    # correct here? Or we could make a special type of Data/Container, that
    # enforces the x, y, index structure?
    new_dict = {}
    for feature in data['x'].shape.keys():
        if feature.startswith('ecg'):
            new_dict[feature] = Data(
                model.predict(data['x'][feature].as_numpy())
            )
        else:
            new_dict[feature] = data['x'][feature]

    processed_ecgs = Container(new_dict)

    new_data = Container(
        {
            'x': processed_ecgs,
            'y': data['y'],
            'index': data['index']
        },
        index=data.index,
        fits_in_memory=True
    )
    return new_data
