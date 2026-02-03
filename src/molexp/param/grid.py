"""Grid search parameter space."""

from __future__ import annotations

from collections.abc import Generator
from itertools import product
from typing import Any

from .base import ParamSpace, Params


class GridSpace(ParamSpace):
    """Grid search parameter space.
    
    Generates all combinations of parameter values using Cartesian product.
    This is equivalent to nested loops over all parameter values.
    
    Example:
        >>> space = GridSpace({
        ...     "d_model": [128, 256],
        ...     "nhead": [4, 8],
        ...     "lr": [1e-4, 5e-4],
        ... })
        >>> for params in space:
        ...     print(params)
        {'d_model': 128, 'nhead': 4, 'lr': 0.0001}
        {'d_model': 128, 'nhead': 4, 'lr': 0.0005}
        {'d_model': 128, 'nhead': 8, 'lr': 0.0001}
        ...
    """
    
    def __init__(self, param_grid: dict[str, list[Any]]) -> None:
        """Initialize grid search space.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values.
                       Each list defines the possible values for that parameter.
        """
        self.param_grid = param_grid
        self._param_names = list(param_grid.keys())
        self._param_values = list(param_grid.values())
        
        # Compute total number of combinations
        self._total = 1
        for values in self._param_values:
            self._total *= len(values)
    
    def __iter__(self) -> Generator[Params, None, None]:
        """Generate all parameter combinations.
        
        Yields:
            Parameter dictionaries, one for each combination
        """
        for combination in product(*self._param_values):
            yield dict(zip(self._param_names, combination))
    
    def __len__(self) -> int:
        """Return total number of combinations.
        
        Returns:
            Number of parameter combinations
        """
        return self._total
    
    def __repr__(self) -> str:
        """String representation."""
        return f"GridSpace({len(self)} combinations)"



