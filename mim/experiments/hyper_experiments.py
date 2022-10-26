from typing import NamedTuple, Any

from mim.experiments.experiments import Experiment
from mim.util.logs import get_logger
from mim.util.util import callable_to_string

log = get_logger("Hyper-experiments")


class HyperExperiment(NamedTuple):
    template: Experiment
    random_seed: int = 42
    strategy: Any = None
    strategy_kwargs: dict = None
    validate_experiment_params: Any = None

    def run(self, action='train'):
        if action != 'train':
            raise ValueError("Can't validate hyper-experiments yet.")
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
            validator=self.validate_experiment_params,
            random_seed=self.random_seed,
            **self.strategy_kwargs
        )

    @property
    def experiments(self):
        return self._searcher().experiments()

    def asdict(self):
        return callable_to_string(self._asdict())

    @property
    def is_trained(self):
        # TODO: Fix this!
        return False


def validate_pool_size(xp_kwargs, minimum_output_size=4):
    try:
        model_kwargs = xp_kwargs['model_kwargs']
    except KeyError:
        return True

    try:
        cnn_kwargs = model_kwargs['cnn_kwargs']
    except KeyError:
        return True

    try:
        mode = xp_kwargs['extractor_kwargs']['features']['ecg_mode']
    except KeyError:
        return True

    shape = 10000 if mode == 'raw' else 1200
    if 'downsample' in cnn_kwargs and cnn_kwargs['downsample']:
        shape //= 2

    num_layers = cnn_kwargs['num_layers']
    if 'pool_sizes' not in cnn_kwargs:
        pool_sizes = num_layers * [cnn_kwargs['pool_size']]
    else:
        pool_sizes = cnn_kwargs['pool_sizes']

    for pool in pool_sizes:
        shape //= pool

    if shape < minimum_output_size:
        return False

    if cnn_kwargs['kernel_last'] >= (shape * pool_sizes[-1]) / 2:
        return False

    return True
