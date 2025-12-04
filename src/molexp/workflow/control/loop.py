"""Loop nodes: Iterative execution patterns."""

from __future__ import annotations

from typing import Any, Callable
from pydantic import BaseModel, Field

from ..node import Node
from ..config import EmptyConfig
from ..registry import register


class LoopConfig(BaseModel):
    """Configuration for basic loop node.
    
    Attributes
    ----------
    iterations : int
        Number of times to repeat
    """
    
    iterations: int = Field(..., ge=1, description="Number of iterations")


@register("control.loop")
class LoopNode(Node[LoopConfig, Any]):
    """Repeat a node a fixed number of times.
    
    Each iteration feeds its output to the next iteration.
    
    Examples
    --------
    >>> body_task = IncrementNode()
    >>> loop_node = LoopNode(initial_value, body_task=body_task)
    >>> result = loop_node(iterations=10)
    """
    
    config_type = LoopConfig
    
    def __init__(
        self,
        initial_value: Any,
        body_task: Node,
        id: str | None = None,
    ):
        """Initialize loop node.
        
        Parameters
        ----------
        initial_value : Any
            Starting value
        body_task : Node
            Node to execute on each iteration
        id : str | None
            Unique identifier
        """
        super().__init__(initial_value, id=id or f"{body_task.id}__loop")
        self.body_task = body_task
    
    def execute(self, initial: Any, config: LoopConfig) -> Any:
        """Execute loop for specified iterations.
        
        Parameters
        ----------
        initial : Any
            Initial value
        config : LoopConfig
            Configuration with iteration count
            
        Returns
        -------
        Any
            Final value after all iterations
        """
        value = initial
        for _ in range(config.iterations):
            if isinstance(value, tuple):
                value = self.body_task(*value)
            else:
                value = self.body_task(value)
        return value


class WhileLoopConfig(BaseModel):
    """Configuration for while loop node.
    
    Attributes
    ----------
    max_iterations : int
        Safety limit to prevent infinite loops
    """
    
    max_iterations: int = Field(
        1000,
        ge=1,
        description="Maximum iterations (safety limit)"
    )


@register("control.while_loop")
class WhileLoopNode(Node[WhileLoopConfig, Any]):
    """Loop until a condition becomes False.
    
    Examples
    --------
    >>> def not_converged(x): return abs(x - 1.0) > 0.01
    >>> body_task = OptimizeStepNode()
    >>> loop_node = WhileLoopNode(
    ...     initial_value,
    ...     condition_fn=not_converged,
    ...     body_task=body_task
    ... )
    >>> result = loop_node(max_iterations=100)
    """
    
    config_type = WhileLoopConfig
    
    def __init__(
        self,
        initial_value: Any,
        condition_fn: Callable[[Any], bool],
        body_task: Node,
        id: str | None = None,
    ):
        """Initialize while loop node.
        
        Parameters
        ----------
        initial_value : Any
            Starting value
        condition_fn : Callable[[Any], bool]
            Function that returns True to continue
        body_task : Node
            Node to execute on each iteration
        id : str | None
            Unique identifier
        """
        super().__init__(initial_value, id=id or f"{body_task.id}__while")
        self.condition_fn = condition_fn
        self.body_task = body_task
    
    def execute(self, initial: Any, config: WhileLoopConfig) -> Any:
        """Execute loop until condition is False.
        
        Parameters
        ----------
        initial : Any
            Initial value
        config : WhileLoopConfig
            Configuration with max_iterations
            
        Returns
        -------
        Any
            Final value when condition becomes False
        """
        value = initial
        iteration = 0
        
        while iteration < config.max_iterations:
            if not self.condition_fn(value):
                break
            
            if isinstance(value, tuple):
                value = self.body_task(*value)
            else:
                value = self.body_task(value)
            
            iteration += 1
        
        if iteration >= config.max_iterations:
            import warnings
            warnings.warn(
                f"WhileLoopNode '{self.id}' reached max iterations ({config.max_iterations})",
                RuntimeWarning
            )
        
        return value


class ForLoopConfig(BaseModel):
    """Configuration for for loop node.
    
    Attributes
    ----------
    iterations : int
        Number of iterations
    """
    
    iterations: int = Field(..., ge=1, description="Number of iterations")


@register("control.for_loop")
class ForLoopNode(Node[ForLoopConfig, Any]):
    """Loop with index access.
    
    The body task receives (current_value, index) as arguments.
    
    Examples
    --------
    >>> def accumulate(state, i): return state + i
    >>> body_task = AccumulateNode()
    >>> loop_node = ForLoopNode(initial_value, body_task=body_task)
    >>> result = loop_node(iterations=10)
    """
    
    config_type = ForLoopConfig
    
    def __init__(
        self,
        initial_value: Any,
        body_task: Node,
        id: str | None = None,
    ):
        """Initialize for loop node.
        
        Parameters
        ----------
        initial_value : Any
            Starting value
        body_task : Node
            Node that takes (value, index) as inputs
        id : str | None
            Unique identifier
        """
        super().__init__(initial_value, id=id or f"{body_task.id}__for")
        self.body_task = body_task
    
    def execute(self, initial: Any, config: ForLoopConfig) -> Any:
        """Execute loop with index.
        
        Parameters
        ----------
        initial : Any
            Initial value
        config : ForLoopConfig
            Configuration with iteration count
            
        Returns
        -------
        Any
            Final value after all iterations
        """
        value = initial
        for i in range(config.iterations):
            value = self.body_task(value, i)
        return value
