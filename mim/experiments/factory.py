from mim.experiments.troponin import MyocardialInfarction


def experiment_from_name(name):
    """
    Put all the classes that inherit from Experiment in here, so that they can
    be easily loaded using only their names.
    """
    name = name.lower()
    if name == 'myocardialinfarction':
        return MyocardialInfarction
    else:
        raise ValueError('No Experiment with name %s ' % name)
