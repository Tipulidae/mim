from mim.experiments.troponin import MyocardialInfarction
from mim.experiments.multiple_ecgs import MultipleECG
from mim.experiments.ab_glucose import ABGlucose
from mim.experiments.random_search import HyperSearch


def experiment_from_name(name):
    """
    Put all the classes that inherit from Experiment in here, so that they can
    be easily loaded using only their names.
    """
    name = name.lower()
    if name == 'myocardialinfarction':
        return MyocardialInfarction
    elif name == 'multipleecg':
        return MultipleECG
    elif name == 'ab':
        return ABGlucose
    elif name == 'hypersearch':
        return HyperSearch
    else:
        raise ValueError(f'No Experiment with name {name}')
