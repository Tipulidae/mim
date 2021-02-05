import math
import random
import itertools
import copy
from enum import Enum
from typing import NamedTuple, Any

import pandas as pd
import tensorflow as tf

import mim.experiments.hyper_parameter as hp
from mim.experiments.experiments import Experiment
from mim.models.simple_nn import basic_cnn2, basic_cnn3, load_keras_model, \
    super_basic_cnn
from mim.fakes.fake_extractors import FakeECG
from mim.extractors.esc_trop import EscTrop
from mim.cross_validation import ChronologicalSplit
from mim.util.logs import get_logger

log = get_logger("Hyper-experiments")


class Searcher:
    def __init__(self, parent_name, parent_base, template,
                 large_score_is_good=True):
        self.parent_name = parent_name
        self.parent_base = parent_base
        self.template = template
        self.large_score_is_good = large_score_is_good

    def search(self):
        raise NotImplementedError()


class Hyperband(Searcher):
    def __init__(self, maximum_resource=200, resource_unit=1, eta=3,
                 random_seed=123, **kwargs):
        super().__init__(**kwargs)
        self.R = maximum_resource
        self.eta = eta
        self.s_max = math.floor(math.log(self.R, self.eta))
        self.B = (self.s_max + 1) * self.R
        self.resource_unit = resource_unit
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
                xp_id_offset=xp_count
            ).search()
            xp_count += n


class SuccessiveHalving(Searcher):
    def __init__(self, num_configurations, minimum_resource,
                 num_brackets=None, eta=3, resource_unit=1,
                 resource_budget=100, random_generator=None,
                 xp_id_offset=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_configurations = num_configurations
        self.minimum_resource = minimum_resource
        self.resource_budget = resource_budget
        self.resource_unit = resource_unit
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
                    experiment.run()
                score = experiment.validation_scores.mean()
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
                ),
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
            args = hp.pick(self.template._asdict(), self.random)
            args['alias'] = self.make_alias(0, i)
            args['parent_base'] = self.parent_base
            args['parent_name'] = self.parent_name
            args['epochs'] = resource_allocation_for_round(
                0, self.minimum_resource, self.resource_unit, self.eta)
            args['initial_epoch'] = 0
            experiments[i] = Experiment(**args)

        return experiments

    def make_alias(self, round, xp_id):
        return f"b{self.bracket}_r{round+1}_x{xp_id + 1 + self.xp_id_offset}"


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
            if xp.is_done:
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
            args = hp.pick(self.template._asdict(), r)
            args['parent_base'] = self.parent_base
            args['parent_name'] = self.parent_name
            args['alias'] = f"xp_{i}"

            yield Experiment(**args)


class HyperExperiment(NamedTuple):
    template: Experiment
    iterations: int = 10
    random_seed: int = 42
    strategy: Any = None
    strategy_kwargs: dict = None

    def run(self):
        log.info(
            f"Running hyper-experiment {self.name} using strategy "
            f"{self.strategy.__name__}"
        )
        searcher = self._searcher()
        searcher.search()

    def _searcher(self):
        return self.strategy(
            template=self.template,
            parent_base=self.__class__.__name__,
            parent_name=self.name,
            **self.strategy_kwargs
        )

    @property
    def experiments(self):
        return self._searcher().experiments()

    @property
    def is_done(self):
        # TODO: Fix this!
        return False


class HyperSearch(HyperExperiment, Enum):
    RS = HyperExperiment(
        template=Experiment(
            description="foo",
            extractor=FakeECG,
            index={
                'n_samples': 1024,
                'shape': (1200, 8),
                'informative_proportion': 0.2,
                'n_classes': 2,
                'random_state': 1111
            },
            model=basic_cnn2,
            building_model_requires_development_data=True,
            optimizer=hp.Choice(['sgd', 'adam']),
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            epochs=0,
            batch_size=hp.Choice([8, 16, 32, 64]),
            model_kwargs={
                'dropout': hp.Float(0.1, 0.8),
                'layers': hp.Choices(
                    data=[{'filters': hp.Choice([8, 16, 32, 64]),
                           'kernel_size': hp.Choice([4, 8, 16, 32])}],
                    k=hp.Int(2, 5)
                ),
                'hidden_layer': hp.Choice(
                    [None, {'dropout': hp.Float(0.1, 0.8)}],
                )
            },
            cv=ChronologicalSplit,
            cv_kwargs={'test_size': 0.3},
            hold_out_size=0
        ),
        random_seed=42,
        strategy=Hyperband,
        strategy_kwargs={
            'maximum_resource': 15,
            'resource_unit': 2
        }
    )

    RS2 = HyperExperiment(
        template=Experiment(
            description="foo",
            extractor=FakeECG,
            index={
                'n_samples': 1024,
                'shape': (1200, 8),
                'informative_proportion': 0.2,
                'n_classes': 2,
                'random_state': 1111
            },
            model=basic_cnn2,
            building_model_requires_development_data=True,
            optimizer=hp.Choice(['sgd', 'adam']),
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            epochs=0,
            batch_size=hp.Choice([8, 16, 32, 64]),
            model_kwargs={
                'dropout': hp.Float(0.1, 0.8),
                'layers': hp.Choices(
                    data=[{'filters': hp.Choice([8, 16, 32, 64]),
                           'kernel_size': hp.Choice([4, 8, 16, 32])}],
                    k=hp.Int(2, 5)
                ),
                'hidden_layer': hp.Choice(
                    [None, {'dropout': hp.Float(0.1, 0.8)}],
                )
            },
            cv=ChronologicalSplit,
            cv_kwargs={'test_size': 0.3},
            hold_out_size=0
        ),
        random_seed=42,
        strategy=Hyperband,
        strategy_kwargs={
            'maximum_resource': 15,
            'resource_unit': 2
        }
    )

    SingleECG = HyperExperiment(
        template=Experiment(
            description="Experiment using variations of a basic CNN for "
                        "predicing MACE using a single ECG record.",
            extractor=EscTrop,
            features={
                'ecg_mode': 'beat',
                'ecgs': ['index']
            },
            index={},
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 0.333
            },
            hold_out_size=0.25,

            model=basic_cnn3,
            building_model_requires_development_data=True,
            optimizer={
                'name': hp.Choice([
                    tf.keras.optimizers.Adam,
                    tf.keras.optimizers.SGD]
                ),
                'kwargs': {
                    'learning_rate':
                        hp.Choice([3e-3, 1e-3, 3e-4, 1e-4])
                }
            },
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            epochs=0,
            batch_size=hp.Choice([32, 64, 128]),
            model_kwargs={
                'layers': hp.Choices(
                    data=[{
                        'filters': hp.Choice([8, 16, 32, 64]),
                        'kernel_size': hp.Choice([4, 8, 16, 32]),
                        'dropout': hp.Float(0.1, 0.8),
                    }],
                    k=hp.Int(2, 5)
                ),
                'pool_size': hp.Int(2, 7),
                'hidden_dropout': hp.Choice(
                    [None, hp.Float(0.1, 0.8)],
                )
            },

        ),
        random_seed=42,
        strategy=Hyperband,
        strategy_kwargs={
            'maximum_resource': 40,
            'resource_unit': 5
        }
    )

    SingleRawECG = SingleECG._replace(
        template=SingleECG.template._replace(
            features={
                'ecg_mode': 'raw',
                'ecgs': ['index']
            },
        )
    )

    LearningRates = HyperExperiment(
        template=Experiment(
            description="Experiment using very simple cnn, testing various "
                        "learning rates and normalization parameters",
            extractor=EscTrop,
            features={
                'ecg_mode': 'beat',
                'ecgs': ['index']
            },
            index={},
            cv=ChronologicalSplit,
            cv_kwargs={
                'test_size': 0.333
            },
            hold_out_size=0.25,
            model=super_basic_cnn,
            building_model_requires_development_data=True,
            optimizer={
                'name': tf.keras.optimizers.Adam,
                'kwargs': {
                    'learning_rate':
                        hp.Choice([3e-3, 1e-3, 3e-4, 1e-4, 3e-5])
                }
            },
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc'],
            epochs=200,
            random_state=hp.Int(0, 1000000000),
            batch_size=hp.Choice([8, 16, 32, 64, 128]),
            model_kwargs={
                'dropout': hp.Choice(
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                )
            },
        ),
        random_seed=42,
        strategy=RandomSearch,
        strategy_kwargs={
            'iterations': 1000
        }
    )
