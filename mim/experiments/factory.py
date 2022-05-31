from importlib import import_module


def import_class(module_name, class_name):
    return getattr(import_module(module_name), class_name)


def experiment_from_name(name):
    """
    Put all the classes that inherit from Experiment in here, so that they can
    be easily loaded using only their names.
    """
    name = name.lower()
    if name == 'esct':
        return import_class('mim.experiments.serial_ecg', 'ESCT')
    elif name == 'ab':
        return import_class('mim.experiments.ab_glucose', 'ABGlucose')
    elif name == 'hypersearch':
        return import_class('mim.experiments.serial_ecg', 'HyperSearch')
    elif name == "genderpredict":
        return import_class('mim.experiments.autoencoders', 'GenderPredict')
    elif name == "article2":
        return import_class('mim.experiments.article2', 'SK1718')
    elif name == 'ptbxl':
        return import_class('mim.experiments.ptbxl', 'ptbxl')
    elif name == 'hyper_ptbxl':
        return import_class('mim.experiments.ptbxl', 'HyperPTBXL')
    else:
        raise ValueError(f'No Experiment with name {name}')
