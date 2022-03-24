from mim.experiments.troponin import MyocardialInfarction
from mim.experiments.serial_ecg import ESCT
from mim.experiments.ab_glucose import ABGlucose
from mim.experiments.hyper_experiments import HyperSearch
from mim.experiments.autoencoders import GenderPredict
from mim.experiments.article2 import SK1718
from mim.experiments.ptbxl import ptbxl


def experiment_from_name(name):
    """
    Put all the classes that inherit from Experiment in here, so that they can
    be easily loaded using only their names.
    """
    name = name.lower()
    if name == 'myocardialinfarction':
        return MyocardialInfarction
    elif name == 'esct':
        return ESCT
    elif name == 'ab':
        return ABGlucose
    elif name == 'hypersearch':
        return HyperSearch
    elif name == "genderpredict":
        return GenderPredict
    elif name == "article2":
        return SK1718
    elif name == 'ptbxl':
        return ptbxl
    else:
        raise ValueError(f'No Experiment with name {name}')
