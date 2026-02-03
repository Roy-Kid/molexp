"""Base classes for parameter spaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

# Type alias for parameter dictionaries
Params = dict[str, Any]


class ParamSpace(ABC):
    """Abstract base class for parameter spaces.
    
    A parameter space defines how to generate parameter combinations
    for hyperparameter search. Subclasses implement different search
    strategies (grid search, random search, etc.).
    """
    
    @abstractmethod
    def __iter__(self) -> Generator[Params, None, None]:
        """Generate parameter combinations.
        
        Yields:
            Parameter dictionaries, each representing one combination
        """
        ...
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of parameter combinations.
        
        Returns:
            Number of combinations (may be infinite for some spaces)
        """
        ...
    
    def count(self) -> int:
        """Count the number of parameter combinations.
        
        Returns:
            Number of combinations
        """
        return len(self)



