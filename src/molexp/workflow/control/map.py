"""Map task: Apply operation to each element of a collection."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..task import Task
from ..plugin.registry import register


class MapConfig(BaseModel):
    """Configuration for MapTask."""

    map_over: str = "items"  # The input key to iterate over


@register("control.map")
class MapTask(Task[MapConfig, list[Any]]):
    """Apply a task to each element of an iterable input.

    This is a control flow task that enables iteration in workflows.
    The workflow engine handles parallelism.

    Examples:
        >>> map_task = MapTask(base_task=some_task, map_over="params")
        >>> result = map_task(ctx=ctx, params=[...], other_input=value)
    """

    config_type = MapConfig
    inputs = {}
    outputs = {}
    replayable = False

    def __init__(
        self,
        base_task: Task,
        /,
        map_over: str = "items",
        static_inputs: list[str] | None = None,
        **config_kwargs: Any
    ):
        """Initialize MapTask.

        Args:
            base_task: The task to apply to each element (positional-only)
            map_over: The input key to iterate over
            static_inputs: Input names that come from workflow (not from items).
                          If None, assumes all base_task inputs come from items.
            **config_kwargs: Additional config arguments
        """
        super().__init__(map_over=map_over, **config_kwargs)
        self.base_task = base_task

        # MapTask inputs = iterable + static inputs
        # Dynamic inputs (from items) are NOT in MapTask.inputs
        self.inputs = {map_over: list}
        if static_inputs:
            for input_name in static_inputs:
                if input_name in base_task.inputs:
                    self.inputs[input_name] = base_task.inputs[input_name]

        self.outputs = base_task.outputs.copy() if base_task.outputs else {}

    def execute(self, ctx=None, **inputs) -> dict[str, Any]:
        """Apply base_task to each element."""
        iterable = inputs.pop(self.config.map_over, [])
        # Remaining inputs are passed to each base_task call
        static_inputs = inputs

        results: list[Any] = []
        for item in iterable:
            if isinstance(item, dict):
                # Merge static inputs with item
                merged_inputs = {**static_inputs, **item}
                result = self.base_task(ctx=ctx, **merged_inputs)
            else:
                result = self.base_task(ctx=ctx, value=item, **static_inputs)
            results.append(result)

        return {"results": results}
