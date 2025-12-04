"""GeneratorNode: Zero inputs to output primitive."""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic
from pydantic import BaseModel

from ..node import Node, CfgT, OutT


class GeneratorNode(Node[CfgT, OutT], Generic[CfgT, OutT]):
    """Primitive for generating output without inputs.
    
    This node type produces output based solely on its configuration.
    Common use cases: data sources, file loaders, constant generators.
    
    Examples:
        >>> class LoadFileConfig(BaseModel):
        ...     path: str
        ...     format: str = "txt"
        ...
        >>> class LoadFileNode(GeneratorNode[LoadFileConfig, str]):
        ...     config_type = LoadFileConfig
        ...     
        ...     def generate(self, config: LoadFileConfig) -> str:
        ...         with open(config.path) as f:
        ...             return f.read()
        ...
        >>> node = LoadFileNode()
        >>> content = node(path="data.txt")
    """
    
    @abstractmethod
    def generate(self, config: CfgT) -> OutT:
        """Generate output based on configuration.
        
        Subclasses must implement this method.
        
        Args:
            config: Validated configuration
            
        Returns:
            Generated output
        """
        raise NotImplementedError
    
    def execute(self, config: CfgT) -> OutT:
        """Execute the generation.
        
        This method calls generate() and can be overridden for
        pre/post-processing if needed.
        
        Args:
            config: Validated configuration
            
        Returns:
            Generated output
        """
        return self.generate(config)
    
    def __call__(self, **config_kwargs) -> OutT:
        """Callable interface for generator nodes.
        
        Overrides Node.__call__ to not expect any inputs.
        
        Args:
            **config_kwargs: Configuration parameters
            
        Returns:
            Generated output
        """
        config = self.config_type(**config_kwargs)
        return self.execute(config)
