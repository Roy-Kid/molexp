"""Conditional node: Branch based on condition."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel

from ..config import EmptyConfig
from ..node import Node
from ..plugin.registry import register


class ConditionalConfig(EmptyConfig):
    """Configuration for ConditionalNode.

    The branching logic is determined by the condition function
    and tasks passed during initialization.
    """

    pass


@register("control.conditional")
class ConditionalNode(Node[ConditionalConfig, Any]):
    """Execute one of two branches based on a runtime condition.

    This node evaluates a condition function and executes either
    the 'then' or 'else' branch, but not both (lazy evaluation).

    Examples
    --------
    >>> def is_positive(x): return x > 0
    >>> then_task = ProcessPositiveNode()
    >>> else_task = ProcessNegativeNode()
    >>> cond_node = ConditionalNode(
    ...     input_value,
    ...     condition_fn=is_positive,
    ...     then_task=then_task,
    ...     else_task=else_task
    ... )
    >>> result = cond_node()
    """

    config_type = ConditionalConfig

    def __init__(
        self,
        condition_input: Any,
        condition_fn: Callable[[Any], bool],
        then_task: Node,
        else_task: Node,
        id: str | None = None,
    ):
        """Initialize conditional node.

        Parameters
        ----------
        condition_input : Any
            Input to evaluate condition on
        condition_fn : Callable[[Any], bool]
            Function that returns True/False
        then_task : Node
            Node to execute if condition is True
        else_task : Node
            Node to execute if condition is False
        id : str | None
            Unique identifier
        """
        super().__init__(condition_input, id=id or "conditional")
        self.condition_fn = condition_fn
        self.then_task = then_task
        self.else_task = else_task

    def execute(self, condition_input: Any) -> Any:
        """Evaluate condition and execute selected branch.

        Parameters
        ----------
        condition_input : Any
            Input value
        config : ConditionalConfig
            Configuration (unused)

        Returns
        -------
        Any
            Result from selected branch
        """
        condition_result = self.condition_fn(condition_input)
        selected_task = self.then_task if condition_result else self.else_task

        # Execute selected branch
        if isinstance(condition_input, tuple):
            return selected_task(*condition_input)
        else:
            return selected_task(condition_input)
