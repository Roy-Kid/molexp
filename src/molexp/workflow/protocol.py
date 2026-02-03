"""Protocol definitions for workflow tasks.

This module defines the structural interface that all workflow tasks must implement,
using Python's typing.Protocol for duck typing without inheritance.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["TaskProtocol"]


@runtime_checkable
class TaskProtocol(Protocol):
    """Structural interface for workflow tasks.
    
    Any class implementing these methods is protocol-compatible for in-process use.
    Persisted workflows still require Task classes with Pydantic configurations.
    
    Example:
        >>> from pydantic import BaseModel
        >>> class MyConfig(BaseModel):
        ...     value: int
    """
    
    def execute(self, ctx: Any | None = None, **inputs: Any) -> dict[str, Any]:
        """Execute the task with given inputs.
        
        Args:
            ctx: Optional runtime context (e.g., RunContext) for accessing
                 directories, logging, asset registration, etc.
            **inputs: Named input values matching the task's input schema
            
        Returns:
            Dictionary of named outputs matching the task's output schema
            
        Example:
            >>> task.execute(input=data, ctx=run_context)
            {'result': <computed_value>}
        """
        ...
    
    def dump(self) -> dict[str, Any]:
        """Serialize task configuration to dictionary.
        
        This method enables task persistence, reproducibility, and inspection.
        The returned dictionary should contain all configuration parameters
        needed to reconstruct the task.
        
        Returns:
            Dictionary representation of task configuration
            
        Example:
            >>> task.dump()
            {'param1': 'value1', 'param2': 42}
        """
        ...
