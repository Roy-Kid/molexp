"""Base Node abstraction for all executable units."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Type, List
from pydantic import BaseModel, ValidationError

# Type variables for configuration and output
CfgT = TypeVar("CfgT", bound=BaseModel)
OutT = TypeVar("OutT")


class Node(Generic[CfgT, OutT], ABC):
    """Base abstraction for all executable units in workflows.
    
    A Node represents a single unit of computation that:
    - Has a unique identifier
    - Accepts typed inputs
    - Produces typed outputs
    - Is configured via a Pydantic model
    - Can be executed standalone or within a workflow
    
    Attributes:
        config_type: Pydantic model class for configuration (must be set by subclasses)
        id: Unique identifier for this node instance
        upstreams: Dependencies (other nodes or constant values)
    """
    
    # Class attribute - must be overridden by subclasses
    config_type: Type[CfgT]
    
    def __init__(self, *upstreams: Any, id: str | None = None) -> None:
        """Initialize node with optional upstream dependencies.
        
        Args:
            *upstreams: Upstream nodes or constant values this node depends on
            id: Unique identifier. If None, uses class name.
        """
        self.id = id or self.__class__.__name__
        self.upstreams: List[Any] = list(upstreams)
    
    @abstractmethod
    def execute(self, *inputs: Any, config: CfgT) -> OutT:
        """Execute the node with given inputs and configuration.
        
        This is the core method that subclasses must implement.
        
        Args:
            *inputs: Input values (from upstream nodes or constants)
            config: Validated configuration instance
            
        Returns:
            Node output
        """
        raise NotImplementedError
    
    def __call__(self, *inputs: Any, **config_kwargs: Any) -> OutT:
        """Callable interface - validates config and executes.
        
        This allows nodes to be called directly from Python code:
        
            result = my_node(input_data, param1="value", param2=42)
        
        Args:
            *inputs: Input values
            **config_kwargs: Configuration parameters (will be validated against config_type)
            
        Returns:
            Node output
            
        Raises:
            ValueError: If config_kwargs don't match the config schema
        """
        try:
            config = self.config_type(**config_kwargs)
        except ValidationError as e:
            raise ValueError(
                f"Invalid configuration for {self.__class__.__name__}: {e}"
            ) from e
        
        return self.execute(*inputs, config=config)
    
    def iter_node_upstreams(self) -> List[Node]:
        """Yield upstream dependencies that are nodes (not constants).
        
        Returns:
            List of upstream nodes
        """
        return [u for u in self.upstreams if isinstance(u, Node)]
    
    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Get JSON schema for this node's configuration.
        
        Returns:
            JSON schema dictionary
        """
        return cls.config_type.model_json_schema()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id!r})"
