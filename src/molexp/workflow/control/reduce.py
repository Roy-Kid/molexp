"""Reduce task: Aggregate collection using a reduction strategy."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..task import Task
from ..plugin.registry import register


class ReduceConfig(BaseModel):
    """Configuration for ReduceTask.

    Attributes:
        method: Reduction method: "sum", "mean", "max", "min", "concat"
    """

    method: Literal["sum", "mean", "max", "min", "concat"] = Field(
        ..., description="Reduction method to apply"
    )


@register("control.reduce")
class ReduceTask(Task[ReduceConfig, Any]):
    """Reduce a collection to a single value using a strategy.

    Configuration (method) must be provided at construction.

    Examples:
        >>> reduce_task = ReduceTask(method="sum")
        >>> result = reduce_task(ctx=ctx, collection=data)
        >>> total = result["result"]
        >>>
        >>> avg_task = ReduceTask(method="mean")
        >>> result = avg_task(ctx=ctx, collection=data)
        >>> average = result["result"]
    """

    config_type = ReduceConfig
    inputs = {"collection": list}
    outputs = {"result": Any}

    def execute(self, ctx=None, **inputs) -> dict[str, Any]:
        """Reduce collection using self.config.method.

        Args:
            ctx: Execution context
            **inputs: Must contain 'collection' (list)

        Returns:
            Dict with 'result' key containing reduced value

        Raises:
            ValueError: If method is unknown
        """
        collection = inputs.get("collection", [])
        data = list(collection)

        if self.config.method == "sum":
            result = sum(data)
        elif self.config.method == "mean":
            result = sum(data) / len(data) if data else 0
        elif self.config.method == "max":
            result = max(data) if data else None
        elif self.config.method == "min":
            result = min(data) if data else None
        elif self.config.method == "concat":
            # Concatenate strings or lists
            if not data:
                result = []
            elif isinstance(data[0], str):
                result = "".join(data)
            else:
                result = []
                for item in data:
                    result.extend(item)
        else:
            raise ValueError(f"Unknown reduce method: {self.config.method}")
        
        return {"result": result}
