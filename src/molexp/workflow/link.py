"""Link abstraction for workflow connections.

This module contains the Link class for representing connections between tasks
in a workflow graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    from .task import Task


class Link(BaseModel):
    """Connection between two tasks in a workflow.

    Links represent dependencies between tasks with explicit or automatic
    output-to-input mappings.

    Attributes:
        source: Source task (Task instance or task ID string)
        target: Target task (Task instance or task ID string)
        mapping: Optional output-to-input mapping. If None, auto-maps matching names.
        status: Link status (default "pending")
    """

    source: str
    target: str
    mapping: dict[str, str] | None = None
    status: str = "pending"

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("source", "target", mode="before")
    @classmethod
    def extract_task_id(cls, v: Any) -> str:
        """Extract task_id from Task instance or return string as-is.

        Args:
            v: Task instance or string task ID

        Returns:
            Task ID as string
        """
        # Import here to avoid circular imports
        from .task import Task

        if isinstance(v, Task):
            return v.task_id
        return v
