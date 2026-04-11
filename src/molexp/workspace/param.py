"""Parameter spaces for experiment hyperparameter search.

Provides:
- ParamSpace: abstract base class
- GridSpace: exhaustive Cartesian product
- UniformSpace: uniform random sampling
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Generator
from itertools import product
from typing import Any

Params = dict[str, Any]


class ParamSpace(ABC):
    """Abstract base class for parameter spaces.

    A parameter space defines how to generate parameter combinations
    for hyperparameter search.
    """

    @abstractmethod
    def __iter__(self) -> Generator[Params, None, None]:
        """Generate parameter combinations."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of parameter combinations."""
        ...

    def count(self) -> int:
        """Alias for len()."""
        return len(self)


class GridSpace(ParamSpace):
    """Exhaustive Cartesian product over all parameter values.

    Example::

        space = GridSpace({"lr": [1e-4, 5e-4], "batch": [32, 64]})
        # yields 4 combinations
    """

    def __init__(self, param_grid: dict[str, list[Any]]) -> None:
        self.param_grid = param_grid
        self._param_names = list(param_grid.keys())
        self._param_values = list(param_grid.values())
        self._total = 1
        for values in self._param_values:
            self._total *= len(values)

    def __iter__(self) -> Generator[Params, None, None]:
        for combination in product(*self._param_values):
            yield dict(zip(self._param_names, combination))

    def __len__(self) -> int:
        return self._total

    def __repr__(self) -> str:
        return f"GridSpace({len(self)} combinations)"


class UniformSpace(ParamSpace):
    """Uniform random sampling from discrete value lists.

    Example::

        space = UniformSpace({"lr": [1e-4, 5e-4, 1e-3]}, n_samples=20, seed=42)
    """

    def __init__(
        self,
        param_values: dict[str, list[Any]],
        n_samples: int,
        seed: int | None = None,
    ) -> None:
        self.param_values = param_values
        self.n_samples = n_samples
        self.seed = seed

    def __iter__(self) -> Generator[Params, None, None]:
        rng = random.Random(self.seed)
        for _ in range(self.n_samples):
            params: Params = {}
            for name, values in self.param_values.items():
                if not isinstance(values, list):
                    raise ValueError(
                        f"Parameter '{name}' must be a list, got {type(values)}"
                    )
                if len(values) == 0:
                    raise ValueError(f"Parameter '{name}' must have at least one value")
                params[name] = rng.choice(values)
            yield params

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return f"UniformSpace({self.n_samples} samples, seed={self.seed})"
