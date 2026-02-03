"""Uniform distribution parameter space."""

from __future__ import annotations

import random
from collections.abc import Generator
from typing import Any

from .base import ParamSpace, Params


class UniformSpace(ParamSpace):
    """Uniform distribution parameter space.
    
    Generates random parameter combinations by uniformly sampling from
    discrete value lists for each parameter.
    
    Example:
        >>> space = UniformSpace({
        ...     "d_model": [128, 256, 512],
        ...     "lr": [1e-4, 5e-4, 1e-3],
        ... }, n_samples=100, seed=42)
        >>> for params in space:
        ...     print(params)
        {'d_model': 256, 'lr': 0.0005}
        ...
    """
    
    def __init__(
        self,
        param_values: dict[str, list[Any]],
        n_samples: int,
        seed: int | None = None,
    ) -> None:
        """Initialize uniform search space.
        
        Args:
            param_values: Dictionary mapping parameter names to lists of possible values.
                         Each value in the list has equal probability of being sampled.
            n_samples: Number of random samples to generate
            seed: Random seed for reproducibility
        """
        self.param_values = param_values
        self.n_samples = n_samples
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
    
    def __iter__(self) -> Generator[Params, None, None]:
        """Generate random parameter combinations.
        
        Yields:
            Parameter dictionaries, one for each random sample
        """
        if self.seed is not None:
            random.seed(self.seed)
        
        for _ in range(self.n_samples):
            params: Params = {}
            for name, values in self.param_values.items():
                if not isinstance(values, list):
                    raise ValueError(
                        f"Parameter '{name}' must have a list of values, "
                        f"got {type(values)}"
                    )
                if len(values) == 0:
                    raise ValueError(f"Parameter '{name}' must have at least one value")
                params[name] = random.choice(values)
            yield params
    
    def __len__(self) -> int:
        """Return number of samples.
        
        Returns:
            Number of random samples
        """
        return self.n_samples
    
    def __repr__(self) -> str:
        """String representation."""
        return f"UniformSpace({self.n_samples} samples, seed={self.seed})"



