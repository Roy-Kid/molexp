from typing import Protocol, Generator
from itertools import combinations, product, islice
import random
from .logging_config import get_logger

logger = get_logger("param")

class Param(dict): ...

class ParamSpace(dict): ...

class ParamSampler(Protocol):

    def sample(self, space: ParamSpace) -> Generator:
        """Sample a parameter set from the given parameter space."""
        ...
    
    def __call__(self, space: ParamSpace) -> Param:
        """return one sampled parameter set per call."""
        return next(self.sample(space))

class CartesianSampler(ParamSampler):
    def sample(self, space: ParamSpace) -> Generator[Param, None, None]:
        """Sample a parameter set from the given parameter space using Cartesian product."""
        from itertools import product
        keys = list(space.keys())
        values = [space[key] for key in keys]
        
        total_combinations = 1
        for v in values:
            total_combinations *= len(v)
        logger.info(f"CartesianSampler generating {total_combinations} parameter combinations")
        
        sampled_values = product(*values)
        for value_combination in sampled_values:
            yield Param({key: value for key, value in zip(keys, value_combination)})

class RandomSampler(ParamSampler):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        logger.debug(f"RandomSampler initialized with {num_samples} samples")

    def sample(self, space: ParamSpace) -> Generator[Param, None, None]:
        """Randomly sample a parameter set from the given parameter space."""
        keys = list(space.keys())
        values = [space[key] for key in keys]
        logger.info(f"RandomSampler generating {self.num_samples} random parameter combinations")
        
        for i in range(self.num_samples):
            sampled_values = [random.choice(v) for v in values]
            logger.debug(f"Generated random sample {i+1}/{self.num_samples}")
            yield Param(zip(keys, sampled_values))

class CombinationSampler(ParamSampler):
    def __init__(self, r: int):
        self.r = r
        logger.debug(f"CombinationSampler initialized with r={r}")

    def sample(self, space: ParamSpace) -> Generator[Param, None, None]:
        """Sample combinations of parameters from the given parameter space."""
        keys = list(space.keys())
        values = [space[key] for key in keys]
        logger.info(f"CombinationSampler generating combinations with r={self.r}")
        for combination in combinations(product(*values), self.r):
            yield Param(zip(keys, combination))