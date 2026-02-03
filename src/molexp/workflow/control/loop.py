"""Loop nodes: Iterative execution patterns."""

from __future__ import annotations

from typing import Any, Callable
import warnings

from pydantic import BaseModel, Field

from ..config import EmptyConfig
from ..task import Task
from ..plugin.registry import register


class LoopConfig(BaseModel):
    """Configuration for basic loop task.

    Attributes
    ----------
    iterations : int
        Number of times to repeat
    """

    iterations: int = Field(..., ge=1, description="Number of iterations")


@register("control.loop")
class LoopTask(Task[LoopConfig, Any]):
    """Repeat a task a fixed number of times.

    Each iteration feeds its output to the next iteration.

    Examples
    --------
    >>> body_task = IncrementTask()
    >>> loop_task = LoopTask(body_task=body_task, iterations=10)
    >>> result = loop_task(ctx=ctx, initial=0)
    >>> final_value = result["result"]
    """

    config_type = LoopConfig
    inputs = {"initial": Any}
    outputs = {"result": Any}
    replayable = False

    def __init__(self, body_task: Task, iterations: int):
        """Initialize loop task.

        Parameters
        ----------
        body_task : Task
            Task to execute on each iteration
        iterations : int
            Number of iterations
        """
        super().__init__(iterations=iterations)
        self.body_task = body_task

    def execute(self, ctx=None, **inputs) -> dict[str, Any]:
        """Execute loop for specified iterations.

        Parameters
        ----------
        ctx : Any | None
            Execution context
        **inputs : Any
            Must contain 'initial' value

        Returns
        -------
        dict[str, Any]
            Dict with 'result' key containing final value
        """
        value = inputs.get("initial")
        
        for _ in range(self.config.iterations):
            if isinstance(value, dict):
                result = self.body_task(ctx=ctx, **value)
            else:
                result = self.body_task(ctx=ctx, value=value)
            # Extract value from result dict
            value = result.get("result", result)
        
        return {"result": value}


class WhileLoopConfig(BaseModel):
    """Configuration for while loop task.

    Attributes
    ----------
    max_iterations : int
        Safety limit to prevent infinite loops
    """

    max_iterations: int = Field(
        1000, ge=1, description="Maximum iterations (safety limit)"
    )


@register("control.while_loop")
class WhileLoopTask(Task[WhileLoopConfig, Any]):
    """Loop until a condition becomes False.

    Examples
    --------
    >>> def not_converged(x): return abs(x - 1.0) > 0.01
    >>> body_task = OptimizeStepTask()
    >>> loop_task = WhileLoopTask(
    ...     condition_fn=not_converged,
    ...     body_task=body_task,
    ...     max_iterations=100
    ... )
    >>> result = loop_task(ctx=ctx, initial=0.0)
    >>> final_value = result["result"]
    """

    config_type = WhileLoopConfig
    inputs = {"initial": Any}
    outputs = {"result": Any}
    replayable = False

    def __init__(
        self,
        body_task: Task,
        condition_fn: Callable[[Any], bool],
        max_iterations: int = 1000,
    ):
        """Initialize while loop task.

        Parameters
        ----------
        body_task : Task
            Task to execute on each iteration
        condition_fn : Callable[[Any], bool]
            Function that returns True to continue
        max_iterations : int
            Safety limit
        """
        super().__init__(max_iterations=max_iterations)
        self.condition_fn = condition_fn
        self.body_task = body_task

    def execute(self, ctx=None, **inputs) -> dict[str, Any]:
        """Execute loop until condition is False.

        Parameters
        ----------
        ctx : Any | None
            Execution context
        **inputs : Any
            Must contain 'initial' value

        Returns
        -------
        dict[str, Any]
            Dict with 'result' key containing final value
        """
        value = inputs.get("initial")
        iteration = 0

        while iteration < self.config.max_iterations:
            if not self.condition_fn(value):
                break

            if isinstance(value, dict):
                result = self.body_task(ctx=ctx, **value)
            else:
                result = self.body_task(ctx=ctx, value=value)
            
            value = result.get("result", result)
            iteration += 1

        if iteration >= self.config.max_iterations:
            warnings.warn(
                f"WhileLoopTask reached max iterations ({self.config.max_iterations})",
                RuntimeWarning,
            )

        return {"result": value}


class ForLoopConfig(BaseModel):
    """Configuration for for loop task.

    Attributes
    ----------
    iterations : int
        Number of iterations
    """

    iterations: int = Field(..., ge=1, description="Number of iterations")


@register("control.for_loop")
class ForLoopTask(Task[ForLoopConfig, Any]):
    """Loop with index access.

    The body task receives value and index as inputs.

    Examples
    --------
    >>> def accumulate(state, i): return state + i
    >>> body_task = AccumulateTask()
    >>> loop_task = ForLoopTask(body_task=body_task, iterations=10)
    >>> result = loop_task(ctx=ctx, initial=0)
    >>> final_value = result["result"]
    """

    config_type = ForLoopConfig
    inputs = {"initial": Any}
    outputs = {"result": Any}
    replayable = False

    def __init__(self, body_task: Task, iterations: int):
        """Initialize for loop task.

        Parameters
        ----------
        body_task : Task
            Task that takes value and index as inputs
        iterations : int
            Number of iterations
        """
        super().__init__(iterations=iterations)
        self.body_task = body_task

    def execute(self, ctx=None, **inputs) -> dict[str, Any]:
        """Execute loop with index.

        Parameters
        ----------
        ctx : Any | None
            Execution context
        **inputs : Any
            Must contain 'initial' value

        Returns
        -------
        dict[str, Any]
            Dict with 'result' key containing final value
        """
        value = inputs.get("initial")
        
        for i in range(self.config.iterations):
            result = self.body_task(ctx=ctx, value=value, index=i)
            value = result.get("result", result)
        
        return {"result": value}
