"""AggregateNode: Multiple inputs to single output primitive."""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar, Generic, List
from pydantic import BaseModel

from ..node import Node, CfgT, OutT

# Input type variable
InT = TypeVar("InT")


class AggregateNode(Node[CfgT, OutT], Generic[CfgT, InT, OutT]):
    """Primitive for aggregating multiple inputs into a single output.
    
    This node type accepts a list of inputs and produces a single output.
    Common use cases: sum, mean, concatenation, reduction operations.
    
    Examples:
        >>> class SumConfig(BaseModel):
        ...     initial: float = 0.0
        ...
        >>> class SumNode(AggregateNode[SumConfig, float, float]):
        ...     config_type = SumConfig
        ...     
        ...     def aggregate(self, inputs: List[float], config: SumConfig) -> float:
        ...         return sum(inputs) + config.initial
        ...
        >>> node = SumNode()
        >>> result = node([1.0, 2.0, 3.0], initial=10.0)  # Returns 16.0
    """
    
    @abstractmethod
    def aggregate(self, inputs: List[InT], config: CfgT) -> OutT:
        """Aggregate multiple inputs into a single output.
        
        Subclasses must implement this method.
        
        Args:
            inputs: List of input values
            config: Validated configuration
            
        Returns:
            Aggregated output
        """
        raise NotImplementedError
    
    def execute(self, *inputs: InT, config: CfgT) -> OutT:
        """Execute the aggregation.
        
        Accepts variable number of inputs and passes them as a list
        to the aggregate() method.
        
        Args:
            *inputs: Variable number of input values
            config: Validated configuration
            
        Returns:
            Aggregated output
        """
        return self.aggregate(list(inputs), config)
