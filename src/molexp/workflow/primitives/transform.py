"""TransformNode: Single input transformation primitive."""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar, Generic
from pydantic import BaseModel

from ..node import Node, CfgT, OutT

# Input type variable
InT = TypeVar("InT")


class TransformNode(Node[CfgT, OutT], Generic[CfgT, InT, OutT]):
    """Primitive for transforming a single input to an output.
    
    This is the most common node type, representing a 1:1 transformation.
    
    Examples:
        >>> class DoubleConfig(BaseModel):
        ...     multiplier: int = 2
        ...
        >>> class DoubleNode(TransformNode[DoubleConfig, int, int]):
        ...     config_type = DoubleConfig
        ...     
        ...     def transform(self, value: int, config: DoubleConfig) -> int:
        ...         return value * config.multiplier
        ...
        >>> node = DoubleNode()
        >>> result = node(5, multiplier=3)  # Returns 15
    """
    
    @abstractmethod
    def transform(self, input: InT, config: CfgT) -> OutT:
        """Transform a single input to an output.
        
        Subclasses must implement this method.
        
        Args:
            input: Input value
            config: Validated configuration
            
        Returns:
            Transformed output
        """
        raise NotImplementedError
    
    def execute(self, input: InT, config: CfgT) -> OutT:
        """Execute the transformation.
        
        This method calls transform() and can be overridden for
        pre/post-processing if needed.
        
        Args:
            input: Input value
            config: Validated configuration
            
        Returns:
            Transformed output
        """
        return self.transform(input, config)
