"""Base Node abstraction for all executable units."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, List, Type, TypeVar

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
    - Is configured via a Pydantic model (fixed at construction)
    - Can be executed standalone or within a workflow

    Configuration is STATIC - it must be provided at construction time
    and cannot be changed during execution.

    Attributes:
        config_type: Pydantic model class for configuration (must be set by subclasses)
        id: Unique identifier for this node instance
        config: Validated configuration instance (set at construction)
        upstreams: Dependencies (other nodes or constant values)
    """

    # Class attribute - must be overridden by subclasses
    config_type: Type[CfgT]

    def __init__(
        self, *upstreams: Any, id: str | None = None, **config_kwargs: Any
    ) -> None:
        """Initialize node with upstream dependencies and static configuration.

        Args:
            *upstreams: Upstream nodes or constant values this node depends on
            id: Unique identifier. If None, uses class name.
            **config_kwargs: Configuration parameters (validated against config_type)

        Raises:
            ValueError: If config_kwargs don't match the config schema
        """
        self.id = id or self.__class__.__name__
        self.upstreams: List[Any] = list(upstreams)

        # Validate and store configuration at construction time
        try:
            self.config: CfgT = self.config_type(**config_kwargs)
        except ValidationError as e:
            raise ValueError(
                f"Invalid configuration for {self.__class__.__name__}: {e}"
            ) from e

    @abstractmethod
    def execute(self, *inputs: Any) -> OutT:
        """Execute the node with given inputs.

        This is the core method that subclasses must implement.
        Configuration is available via self.config.

        Args:
            *inputs: Input values (from upstream nodes or constants)

        Returns:
            Node output
        """
        raise NotImplementedError

    def __call__(self, *inputs: Any) -> OutT:
        """Callable interface - executes with stored configuration.

        This allows nodes to be called directly from Python code:

            node = MyNode(param1="value", param2=42)
            result = node(input_data)

        Args:
            *inputs: Input values

        Returns:
            Node output
        """
        return self.execute(*inputs)

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

    def get_config_dict(self) -> dict[str, Any]:
        """Get configuration as dictionary for serialization.

        Returns:
            Configuration dictionary
        """
        return self.config.model_dump()

    @classmethod
    def from_config_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> Node:
        """Create node instance from configuration dictionary.

        Args:
            config_dict: Configuration parameters as dictionary
            **kwargs: Additional arguments (id, upstreams, etc.)

        Returns:
            New node instance
        """
        return cls(**config_dict, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id!r}, config={self.config!r})"
