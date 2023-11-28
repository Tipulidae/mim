import copy
import itertools
import math
import random

import numpy as np
import pandas as pd

from mim.experiments import hyper_parameter as hp
from mim.experiments.experiments import Experiment
from mim.models.load import load_keras_model
from mim.util.logs import get_logger

log = get_logger("Search Strategies")


class Searcher:
    def __init__(self, parent_name, parent_base, template,
                 validator=None,
                 large_score_is_good=True):
        self.parent_name = parent_name
        self.parent_base = parent_base
        self.template = template
        self.large_score_is_good = large_score_is_good
        self.validator = validator or (lambda _: True)
        self.max_iterations = 1000

    def generate_valid_xp_args(self, random_generator):
        args = None
        for i in range(self.max_iterations):
            args = hp.pick(self.template._asdict(), random_generator)
            if self.validator(args):
                log.debug(f'Found valid arguments after {i+1} attempts.')
                break

        if args is None:
            raise Exception(
                f'No valid experiment settings were found in '
                f'{self.max_iterations} iterations!'
            )
        return args

    def search(self):
        raise NotImplementedError()


class Hyperband(Searcher):
    def __init__(self, maximum_resource=200, resource_unit=1, eta=3,
                 random_seed=123, iterations=1, **kwargs):
        super().__init__(**kwargs)
        self.R = maximum_resource
        self.eta = eta
        self.s_max = math.floor(math.log(self.R, self.eta))
        self.B = (self.s_max + 1) * self.R
        self.resource_unit = resource_unit
        self.iterations = iterations
        self.random = random.Random(random_seed)

    def search(self):
        total_resource_budget = self.R * (self.s_max + 1) ** 2
        epoch_budget = self.resource_unit * total_resource_budget
        log.debug(
            f"Running Hyperband with a total resource budget of "
            f"{total_resource_budget} units, corresponding to {epoch_budget} "
            f"epochs. Hyperband will run {self.s_max + 1} brackets of "
            f"SuccessiveHalving, each of which has a maximum budget of "
            f"{total_resource_budget/(self.s_max + 1)} units, or "
            f"{epoch_budget/(self.s_max + 1)} epochs.\n"
        )

        xp_count = 0
        for it in range(self.iterations):
            log.debug(f'Iteration {it+1} / {self.iterations} of hyperband.')
            for s in range(self.s_max, -1, -1):
                n = math.ceil((self.B / self.R) * (self.eta ** s) / (s + 1))
                r = self.R * self.eta ** (-s)
                SuccessiveHalving(
                    parent_name=self.parent_name,
                    parent_base=self.parent_base,
                    template=self.template,
                    num_configurations=n,
                    minimum_resource=r,
                    resource_unit=self.resource_unit,
                    num_brackets=s+1,
                    eta=self.eta,
                    resource_budget=self.B,
                    random_generator=self.random,
                    large_score_is_good=self.large_score_is_good,
                    xp_id_offset=xp_count,
                    iteration=it
                ).search()
                xp_count += n


class SuccessiveHalving(Searcher):
    def __init__(self, num_configurations, minimum_resource,
                 num_brackets=None, eta=3, resource_unit=1,
                 resource_budget=100, random_generator=None,
                 xp_id_offset=0, iteration=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_configurations = num_configurations
        self.minimum_resource = minimum_resource
        self.resource_budget = resource_budget
        self.resource_unit = resource_unit
        self.iteration = iteration
        self.eta = eta
        self.xp_id_offset = xp_id_offset
        if num_brackets is None:
            self.bracket = calculate_number_of_brackets(
                self.resource_budget,
                self.num_configurations,
                self.minimum_resource
            ) + 1
        else:
            self.bracket = num_brackets

        if random_generator is None:
            self.random = random.Random(123)
        else:
            self.random = random_generator

    def search(self):
        log.debug(
            f"Starting SuccessiveHalving with a total resource budget of "
            f"{self.resource_budget} units, corresponding to "
            f"{self.resource_budget * self.resource_unit} epochs. "
            f"SuccessiveHalving will run {self.bracket} rounds of "
            f"experiments, each of which has a maximum budget of "
            f"{self.resource_budget/self.bracket} units, or "
            f"{self.resource_budget * self.resource_unit / self.bracket} "
            f"epochs."
        )

        experiments = self.init_experiments()
        for i in range(self.bracket):
            allocated_resources = resource_allocation_for_round(
                i, self.minimum_resource, self.resource_unit, self.eta
            )
            log.debug(
                f'Bracket {self.bracket}, round {i+1}: '
                f'n={len(experiments)}, resources={allocated_resources}, '
                f'total={len(experiments) * allocated_resources}'
            )

            scores = {}
            for name, experiment in experiments.items():
                # Now run the experiment and find the loss!
                if not experiment.is_done:
                    s = pd.Series(
                        hp.flatten(experiment.asdict()),
                        name=experiment.name
                    )
                    log.info(f"Experiment {experiment.name} with kwargs\n"
                             f"{s}\n\n")
                    experiment.conduct()
                score = np.mean(experiment.validation_scores)
                scores[name] = score

            experiments = self.top_k(experiments, scores, i)

    def top_k(self, experiments, losses, i):
        k = math.floor(self.num_configurations / (self.eta ** (i + 1)))

        new_experiments = {}
        # reverse=True means values are sorted in descending order (from
        # large to small)
        k_best_experiment_ids = itertools.islice(
            sorted(
                losses,
                key=lambda x: losses[x],
                reverse=self.large_score_is_good
            ),
            k
        )

        for xp_id in k_best_experiment_ids:
            xp = experiments[xp_id]
            new_kwargs = copy.copy(xp.model_kwargs)
            new_kwargs['base_path'] = xp.base_path
            new_experiments[xp_id] = xp._replace(
                alias=self.make_alias(i+1, xp_id),
                epochs=resource_allocation_for_round(
                    i+1, self.minimum_resource, self.resource_unit, self.eta
                ) + xp.epochs,
                initial_epoch=xp.epochs,
                model=load_keras_model,
                model_kwargs=new_kwargs,
                skip_compile=True,
                building_model_requires_development_data=False,
            )

        return new_experiments

    def init_experiments(self):
        experiments = {}
        for i in range(self.num_configurations):
            args = self.generate_valid_xp_args(self.random)
            args['alias'] = self.make_alias(0, i)
            args['parent_base'] = self.parent_base
            args['parent_name'] = self.parent_name
            args['epochs'] = resource_allocation_for_round(
                0, self.minimum_resource, self.resource_unit, self.eta)
            args['initial_epoch'] = 0
            experiments[i] = Experiment(**args)

        return experiments

    def make_alias(self, round, xp_id):
        return f"i{self.iteration + 1}_b{self.bracket}_r" \
               f"{round+1}_x{xp_id + 1 + self.xp_id_offset}"


def resource_allocation_for_round(i, minimum_resource, resource_unit=1, eta=3):
    # Using round here might cause SuccessiveHalving to slightly exceed
    # resource budget
    return math.floor(resource_unit * minimum_resource * eta ** i)


def calculate_number_of_brackets(B, n, r):
    return math.ceil(B / (n * r) - 1)


class RandomSearch(Searcher):
    def __init__(self, restart=False, iterations=10,
                 random_seed=123, **kwargs):
        super().__init__(**kwargs)
        self.restart = restart
        self.iterations = iterations
        self.random_seed = random_seed

    def search(self):
        for xp in self.experiments():
            if xp.is_trained:
                continue

            s = pd.Series(hp.flatten(xp.asdict()), name=xp.name)
            log.info(f"Experiment {xp.name} with kwargs\n"
                     f"{s}\n\n")

            xp.run()

    def experiments(self):
        # For more complicated search strategies, this generator might have
        # to depend on the results of previous experiments. Therefore, using
        # it to, for example, check which experiments are done might not be
        # very well advised, because it would necessarily depend on the state
        # of the searcher itself.....
        # And this is in fact a pretty strong argument in favor of actually
        # saving the searcher progress in a file in the first place!
        r = random.Random(self.random_seed)

        for i in range(self.iterations):
            args = self.generate_valid_xp_args(r)
            args['parent_base'] = self.parent_base
            args['parent_name'] = self.parent_name
            args['alias'] = f"xp_{i}"
            yield Experiment(**args)


class EnsembleRandomSearch(Searcher):
    def __init__(
            self, iterations_per_bracket=None, models_per_bracket=None,
            random_seed=123, **kwargs):
        super().__init__(**kwargs)
        self.random_seed = random_seed
        self.iterations_per_bracket = iterations_per_bracket
        self.models_per_bracket = models_per_bracket

    def search(self):
        experiments = ...
        for _, (splits, num_models) in enumerate(zip(
                range(self.iterations_per_bracket),
                range(self.models_per_bracket))):

            scores = {}
            for name, experiment in experiments.items():
                num_splits_done = experiment.num_splits_done
                if num_splits_done < splits:
                    experiment.run(
                        action='train',
                        restart=False,
                        splits_to_do=splits-num_splits_done
                    )
                scores[name] = self.calculate_xp_score(experiment)

            experiments = self.top_k(num_models, experiments, scores)

    def calculate_xp_score(self, experiment):
        # Some metric that depends on how jagged the loss history is and how
        # well the model performed on the validation set
        return 0

    def top_k(self, num_models, experiments, scores):
        # Sort the scores and find the best num_models. Or something.
        #
        pass
