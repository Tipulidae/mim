import argparse
import re
from importlib import import_module

import silence_tensorflow.auto  # noqa: F401

from mim.util.logs import get_logger
from mim.cache import settings

log = get_logger("Run")


def experiment_from_name(name):
    parts = name.split('.')
    class_name = parts[-1]
    module_name = 'projects.' + '.'.join(parts[:-1])
    return getattr(import_module(module_name), class_name)


def run_experiments(experiments, continue_on_error=False, action='train',
                    restart=False):
    """
    Run all experiments and save the results to disk.

    :param experiments: List of experiments to run
    :param continue_on_error: Whether to continue running experiments even
    if one should fail.
    :param action: Either 'train' or 'test'. Whether to train (and validate)
    the models, or to load a trained model and evaluate it on the test-set.
    """
    for experiment in experiments:
        try:
            experiment.run(action=action, restart=restart)
        except Exception as e:
            log.error(
                f'Something went wrong with task {experiment.name}! '
                f'Oh no! :('
            )
            if not continue_on_error:
                raise e

    log.info('Everything looks good! :)')


if __name__ == '__main__':
    description = """
    An experiment is a recipe that specifies a model, parameters and data.
    Each experiment has a unique name, and is listed in one of the project
    modules. Use this function to either run (train + validate) an experiment,
    or to evaluate it (load a previously trained model and evaluate it on the
    test set). The action parameter specifies whether to train or test the
    experiment.

    Example usage: \n
    'python -m mim.run ddsa.experiments.DDSA -p CNN1_1L' --action test\n
    will evaluate all experiments that starts with CNN1_1L in the
    DDSA experiment enum in the projects.ddsa.experiment module.

    Type --help for more info.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        'base',
        help='name of the base experiment to run. Example: '
             'serial_ecgs.experiments.ESCT'
    )
    parser.add_argument(
        '-a', '--action', type=str, choices=['train', 'test'],
        help='whether to train (and validate) or test the models. '
             'Can only test experiments that are done training. '
             'Default is train.',
        default='train'
    )

    parser.add_argument(
        '-r', '--rerun',
        help='rerun experiments that already have a result. Partially '
             'completed experiments will restart from scratch.',
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
        '-x', '--xps',
        help='comma separated list of experiments to run (if not specified, '
             'run all experiments)',
    )
    parser.add_argument(
        '-p', '--pattern',
        help='run experiments that matches the given pattern '
             '(regular expression)',
    )
    parser.add_argument(
        '-c', '--cache',
        help="Set cache strictness: off = don't cache, safe = invalidate "
             "cache if there are uncommitted files or if commits or "
             "conda environment differs, unsafe = allow different commits "
             "and environments, stupid = never invalidate cache. Caching can "
             "speed up your code, but if used carelessly, can lead to bugs "
             "that are difficult to detect. Use at your own peril.",
        type=str,
        choices=['off', 'safe', 'unsafe', 'stupid'],
        default='off'
    )

    args = parser.parse_args()

    xps_done = []
    xps_to_continue = []
    xps_to_run = []
    xps_to_rerun = []
    xps_not_conducted = []

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

    if args.action == 'train':
        for xp in xps_to_consider:
            if xp.is_trained:
                xps_done.append(xp)
            elif xp.is_partial:
                xps_to_continue.append(xp)
            else:
                xps_to_run.append(xp)
    elif args.action == 'test':
        for xp in xps_to_consider:
            if xp.is_evaluated:
                xps_done.append(xp)
            elif xp.is_trained:
                xps_to_run.append(xp)
            else:
                xps_not_conducted.append(xp)

    if args.rerun:
        xps_to_rerun = xps_done
        xps_to_rerun.extend(xps_to_continue)
        xps_to_continue = []
        xps_done = []

    for xp in xps_done:
        log.info(f'{xp.name} has status DONE')
    for xp in xps_not_conducted:
        log.info(f'{xp.name} has status NOT CONDUCTED')
    for xp in xps_to_run:
        log.info(f'{xp.name} has status RUN')
    for xp in xps_to_continue:
        log.info(f'{xp.name} has status CONTINUE')
    for xp in xps_to_rerun:
        log.info(f'{xp.name} has status RE-RUN')

    log.info(f'{len(xps_done)} experiments has status DONE')
    log.info(f'{len(xps_to_run)} experiments has status RUN')
    log.info(f'{len(xps_to_continue)} experiments has status CONTINUE')
    log.info(f'{len(xps_to_rerun)} experiments has status RE-RUN')
    if xps_not_conducted:
        log.info(f"{len(xps_not_conducted)} experiments has status "
                 f"NOT CONDUCTED and can't be evaluated.")

    all_xps_to_run = xps_to_run + xps_to_rerun

    if args.cache == 'off':
        settings.enabled = False
        log.info("Cache setting: off")
    else:
        settings.enabled = True
        settings.set_level(args.cache)
        log.info(f"Cache setting: {args.cache}")

    if not args.suppress:
        answer = input(
            f'Continue running ({args.action}ing) {len(all_xps_to_run)} '
            f'experiments? (y/n)')
        if answer.lower() != 'y':
            exit(0)

    run_experiments(all_xps_to_run, continue_on_error=args.force,
                    action=args.action, restart=args.rerun)
