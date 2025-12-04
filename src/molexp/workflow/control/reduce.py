"""Reduce node: Aggregate collection using a reduction strategy."""

from __future__ import annotations

from typing import Any, Iterable, Literal
from pydantic import BaseModel, Field

from ..node import Node
from ..registry import register


class ReduceConfig(BaseModel):
    """Configuration for ReduceNode.
    
    Attributes:
        method: Reduction method: "sum", "mean", "max", "min", "concat"
    """
    
    method: Literal["sum", "mean", "max", "min", "concat"] = Field(
        ...,
        description="Reduction method to apply"
    )


@register("control.reduce")
class ReduceNode(Node[ReduceConfig, Any]):
    """Reduce a collection to a single value using a strategy.
    
    Examples:
        >>> reduce_node = ReduceNode(source_collection)
        >>> total = reduce_node(method="sum")
        >>> average = reduce_node(method="mean")
    """
    
    config_type = ReduceConfig
    
    def execute(self, collection: Iterable[Any], config: ReduceConfig) -> Any:
        """Reduce collection using specified method.
        
        Args:
            collection: Collection to reduce
            config: Configuration specifying reduction method
            
        Returns:
            Reduced value
            
        Raises:
            ValueError: If method is unknown
        """
        data = list(collection)
        
        if config.method == "sum":
            return sum(data)
        elif config.method == "mean":
            return sum(data) / len(data) if data else 0
        elif config.method == "max":
            return max(data) if data else None
        elif config.method == "min":
            return min(data) if data else None
        elif config.method == "concat":
            # Concatenate strings or lists
            if not data:
                return []
            if isinstance(data[0], str):
                return "".join(data)
            else:
                result = []
                for item in data:
                    result.extend(item)
                return result
        else:
            raise ValueError(f"Unknown reduce method: {config.method}")
