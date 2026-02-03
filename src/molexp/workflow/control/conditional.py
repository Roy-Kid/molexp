"""Conditional task: Branch based on condition."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel

from ..config import EmptyConfig
from ..task import Task
from ..plugin.registry import register


class ConditionalConfig(EmptyConfig):
    """Configuration for ConditionalTask.

    The branching logic is determined by the condition function
    and tasks passed during initialization.
    """

    pass


@register("control.conditional")
class ConditionalTask(Task[ConditionalConfig, Any]):
    """Execute one of two branches based on a runtime condition.

    This task evaluates a condition function and executes either
    the 'then' or 'else' branch, but not both (lazy evaluation).

    Examples
    --------
    >>> def is_positive(x): return x > 0
    >>> then_task = ProcessPositiveTask()
    >>> else_task = ProcessNegativeTask()
    >>> cond_task = ConditionalTask(
    ...     condition_fn=is_positive,
    ...     then_task=then_task,
    ...     else_task=else_task
    ... )
    >>> result = cond_task(ctx=ctx, condition_input=value)
    >>> output = result["result"]
    """

    config_type = ConditionalConfig
    inputs = {"condition_input": Any}
    outputs = {"result": Any}
    replayable = False

    def __init__(
        self,
        condition_fn: Callable[[Any], bool],
        then_task: Task,
        else_task: Task,
    ):
        """Initialize conditional task.

        Parameters
        ----------
        condition_fn : Callable[[Any], bool]
            Function that returns True/False
        then_task : Task
            Task to execute if condition is True
        else_task : Task
            Task to execute if condition is False
        """
        super().__init__()
        self.condition_fn = condition_fn
        self.then_task = then_task
        self.else_task = else_task

    def execute(self, ctx=None, **inputs) -> dict[str, Any]:
        """Evaluate condition and execute selected branch.

        Parameters
        ----------
        ctx : Any | None
            Execution context
        **inputs : Any
            Must contain 'condition_input'

        Returns
        -------
        dict[str, Any]
            Dict with 'result' key from selected branch
        """
        condition_input = inputs.get("condition_input")
        
        condition_result = self.condition_fn(condition_input)
        selected_task = self.then_task if condition_result else self.else_task

        # Execute selected branch
        if isinstance(condition_input, dict):
            result = selected_task(ctx=ctx, **condition_input)
        else:
            result = selected_task(ctx=ctx, input=condition_input)
        
        return {"result": result}
