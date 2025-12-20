"""Reduce node: Aggregate collection using a reduction strategy."""

from __future__ import annotations

from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from ..node import Node
from ..plugin.registry import register


class ReduceConfig(BaseModel):
    """Configuration for ReduceNode.

    Attributes:
        method: Reduction method: "sum", "mean", "max", "min", "concat"
    """

    method: Literal["sum", "mean", "max", "min", "concat"] = Field(
        ..., description="Reduction method to apply"
    )


@register("control.reduce")
class ReduceNode(Node[ReduceConfig, Any]):
    """Reduce a collection to a single value using a strategy.

    Configuration (method) must be provided at construction.

    Examples:
        >>> reduce_node = ReduceNode(source_collection, method="sum")
        >>> total = reduce_node(data)
        >>>
        >>> avg_node = ReduceNode(source_collection, method="mean")
        >>> average = avg_node(data)
    """

    config_type = ReduceConfig

    def execute(self, collection: Iterable[Any]) -> Any:
        """Reduce collection using self.config.method.

        Args:
            collection: Collection to reduce

        Returns:
            Reduced value

        Raises:
            ValueError: If method is unknown
        """
        data = list(collection)

        if self.config.method == "sum":
            return sum(data)
        elif self.config.method == "mean":
            return sum(data) / len(data) if data else 0
        elif self.config.method == "max":
            return max(data) if data else None
        elif self.config.method == "min":
            return min(data) if data else None
        elif self.config.method == "concat":
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
            raise ValueError(f"Unknown reduce method: {self.config.method}")
