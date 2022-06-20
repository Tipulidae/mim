import argparse
import re
from importlib import import_module

import silence_tensorflow.auto  # noqa: F401

from mim.util.logs import get_logger

log = get_logger("Run")


def experiment_from_name(name):
    parts = name.split('.')
    class_name = parts[-1]
    module_name = 'projects.' + '.'.join(parts[:-1])
    return getattr(import_module(module_name), class_name)


def run_experiments(experiments, continue_on_error=False):
    """
    Run all experiments and save the results to disk.

    :param experiments: List of experiments to run
    :param continue_on_error: Whether to continue running experiments even
    if one should fail.
    """
    for experiment in experiments:
        try:
            experiment.run()
        except Exception as e:
            log.error(
                f'Something went wrong with task {experiment.name}! '
                f'Oh no! :('
            )
            if not continue_on_error:
                raise e

    log.info('Everything looks good! :)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run (or re-run) experiments from enum specified in base "
                    "argument. Experiment is expected to be in the projects "
                    "module. Example: \n "
                    "'python -m mim.run ddsa.experiment.DDSA -p CNN1_1L' \n "
                    "will run all experiments that starts with CNN1_1L in the "
                    "DDSA experiment enum in the projects.ddsa.experiment "
                    "module."
    )
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
        help='name of the base experiment to run. Example: '
             'serial_ecgs.experiments.ESCT'
    )
    parser.add_argument(
        '-x', '--xps',
        help='comma separated list of experiments to run (if not specified, '
             'run all experiments)',
    )
    parser.add_argument(
        '-p', '--pattern',
        help='run experiments that matches the given pattern '
             '(regular expression)',
    )

    args = parser.parse_args()

    xps_done = []
    xps_todo = []
    xps_to_rerun = []

    base_xp_enum = experiment_from_name(args.base)

    if args.xps:
        xps_to_consider = [base_xp_enum[xp_name.strip()] for xp_name in
                           args.xps.split(',')]
    elif args.pattern:
        p = re.compile(args.pattern)
        xps_to_consider = list(filter(
            lambda xp: p.match(xp.name),
            list(base_xp_enum)
        ))
    else:
        xps_to_consider = list(base_xp_enum)

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
