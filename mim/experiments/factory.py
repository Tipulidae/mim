from mim.experiments.troponin import MyocardialInfarction
from mim.experiments.ptbxl import PtbxlMi


def experiment_from_name(name):
    """
    Put all the classes that inherit from Experiment in here, so that they can
    be easily loaded using only their names.
    """
    name = name.lower()
    if name == 'myocardialinfarction':
        return MyocardialInfarction
    elif name == 'ptbxlmi':
        return PtbxlMi
    else:
        raise ValueError(f'No Experiment with name {name}')
