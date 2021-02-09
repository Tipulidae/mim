import argparse
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from mim.experiments.experiments import Experiment
from .factory import experiment_from_name
from mim.util.metadata import Metadata
from mim.util.logs import get_logger

log = get_logger("Run")


def run_experiments(experiments, continue_on_error=False):
    """
    Run all experiments in xps and save the results to disk.

    :param experiments: List of experiments to run
    :param continue_on_error: Whether to continue running experiments even
    if one should fail.
    """
    for experiment in experiments:
        try:
            pd.to_pickle(
                run_one_experiment(experiment),
                experiment.result_path
            )
        except Exception as e:
            log.error(
                f'Something went wrong with task {experiment.name}! '
                f'Oh no! :('
            )
            if not continue_on_error:
                raise e

    log.info('Everything looks good! :)')


def run_one_experiment(experiment: Experiment):
    """
    Generate the data from an experiment, split it into training and
    testing sets according to the prescribed cross-validation technique,
    train the model and save all the results in a dictionary.

    :param experiment: An experiment from mim, of type Experiment or
    Robustness.
    :return: dictionary containing results of the cross-validation. Contains
    times taken for training and prediction in each step, train and test
    scores, predictions and correct targets. Also contains feature
    importance for each split if the model supports it, as well as biases
    and contributions from tree-interpreter if applicable. Also contains
    metadata report.
    """
    log.info(f'Computing validation scores for {experiment.name}: '
             f'{experiment.description}')
    t = time()

    data_provider = experiment.get_data()
    feature_names = None  # AB: What role does this play?

    results = {
        'fit_time': [],
        'score_time': [],
        'test_score': [],
        'feature_importance': [],
        'predictions': [],
        'targets': [],
        'feature_names': feature_names,
        'train_score': [],
        'history': []
    }

    # for train, validation in tqdm(cross_validation.split(data)):

    # Extend below to allow for CV instead, parametrize which subset (train,
    # val etc, as well as K)
    for train, validation in tqdm(data_provider.split()):
        result = _validate(
            train,
            validation,
            experiment.get_model(train, validation),
            experiment.scoring
        )

        _update_results(results, result)

    _finish_results(results)
    results['metadata'] = Metadata().report()

    log.info(f'Finished computing scores for {experiment.name} in '
             f'{time()-t}s. ')
    return results


def _validate(train, val, model, scoring):
    t0 = time()
    log.debug('\n\nFitting classifier...')
    history = model.fit(train, validation_data=val)
    fit_time = time() - t0

    prediction = model.predict(val['x'])
    score_time = time() - fit_time - t0

    train_score = scoring(
        train['y'].as_numpy,
        model.predict(train['x'])['prediction']
    )

    y_val = val['y'].as_numpy
    test_score = scoring(y_val, prediction['prediction'])
    log.debug(f'test score: {test_score}, train score: {train_score}')

    try:
        feature_importance = model.model.feature_importances_
    except AttributeError:
        feature_importance = None

    return {
        'predictions': prediction,
        'train_score': train_score,
        'test_score': test_score,
        'feature_importance': feature_importance,
        'fit_time': fit_time,
        'score_time': score_time,
        'targets': y_val,
        'history': history
    }


def _update_results(results, result):
    for key, value in result.items():
        results[key].append(value)


def _finish_results(results):
    predictions = results['predictions']
    new_predictions = {}
    for prediction in predictions:
        for key, value in prediction.items():
            if key not in new_predictions:
                new_predictions[key] = []
            new_predictions[key].append(value)

    results['predictions'] = new_predictions

    results['fit_time'] = np.array(results['fit_time'])
    results['score_time'] = np.array(results['score_time'])
    results['test_score'] = np.array(results['test_score'])
    results['feature_importance'] = np.array(results['feature_importance'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--rerun',
        help='rerun experiments that already have a result',
        action='store_true'
    )
    parser.add_argument(
        '-s', '--suppress',
        help='suppress prompt - runs the experiments without asking first',
        action='store_true'
    )
    parser.add_argument(
        '-f', '--force',
        help='force the program to continue running other experiments even if '
             'one should fail',
        action='store_true'
    )
    parser.add_argument(
        'base',
        help='name of the base experiment to run'
    )
    parser.add_argument(
        '-x', '--xps',
        help='comma separated list of experiments to run (if not specified, '
             'run all experiments)',
    )

    args = parser.parse_args()

    xps_done = []
    xps_todo = []
    xps_to_rerun = []

    if args.xps:
        base = experiment_from_name(args.base)
        xps_to_consider = [base[xp_name.strip()] for xp_name in
                           args.xps.split(',')]
    else:
        xps_to_consider = list(experiment_from_name(args.base))

    for xp in xps_to_consider:
        if xp.is_done:
            xps_done.append(xp)
        else:
            xps_todo.append(xp)

    if args.rerun:
        xps_to_rerun = xps_done
        xps_done = []

    for xp in xps_done:
        log.info(f'{xp.name} has status DONE')
    for xp in xps_todo:
        log.info(f'{xp.name} has status NOT STARTED')
    for xp in xps_to_rerun:
        log.info(f'{xp.name} has status RERUN')

    log.info(f'{len(xps_done)} experiments has status DONE')
    log.info(f'{len(xps_todo)} experiments has status NOT STARTED')
    log.info(f'{len(xps_to_rerun)} experiments has status RERUN')

    all_xps_to_run = xps_todo + xps_to_rerun

    if not args.suppress:
        answer = input(
            f'Continue running {len(all_xps_to_run)} experiments? (y/n)')
        if answer.lower() != 'y':
            exit(0)

    run_experiments(all_xps_to_run, continue_on_error=args.force)
