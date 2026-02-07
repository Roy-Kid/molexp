"""Link abstraction for workflow connections and channel configuration.

This module provides the Link class for defining dependencies and message channels
between tasks in a workflow graph. Links serve dual purposes:

1. **Dependency Specification**: Define execution order for batch workflows
2. **Channel Configuration**: Configure message passing buffers for actor workflows

For batch tasks, Links specify data flow via output-to-input mappings.
For actor tasks, Links define asyncio.Queue channels with configurable buffer sizes
and explicit channel name routing.

Example:
    Basic dependency link::

        link = Link(source='task_a', target='task_b')

    Actor channel with buffer::

        link = Link(
            source='generator_actor',
            target='processor_actor',
            buffer_size=200,
            mapping={'data': 'input_data'}
        )

    Creating workflow with links::

        workflow = Workflow.from_tasks(
            tasks=[actorA, actorB, actorC],
            links=[
                Link(source=actorA, target=actorB, mapping={'out': 'in'}),
                Link(source=actorB, target=actorC, buffer_size=50)
            ]
        )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    from .task import Task


class Link(BaseModel):
    """Connection between two tasks in a workflow.

    Links represent dependencies between tasks with explicit or automatic
    output-to-input mappings. For Actor workflows, Links also define message
    channels with configurable buffering.

    Attributes:
        source: Source task (Task instance or task ID string)
        target: Target task (Task instance or task ID string)
        mapping: Optional output-to-input mapping. If None, auto-maps matching names.
        buffer_size: Buffer size for actor channels (default: 100).
                    Used when link involves Actors. Larger buffers reduce backpressure
                    but use more memory.
        channels: Optional channel name mapping for actors. Maps source channel names
                 to target channel names. If None, assumes matching names.
                 Example: {'out': 'in'} means source.emit('out') -> target.receive('in')
        status: Link status (default "pending")

    Examples:
        Basic link:
            >>> Link('task_a', 'task_b')

        With buffer size:
            >>> Link('actor_a', 'actor_b', buffer_size=200)

        With channel mapping:
            >>> Link('generator', 'processor', channels={'data': 'input_data'})
    """

    source: str
    target: str
    mapping: dict[str, str] | None = None
    buffer_size: int = 100
    channels: dict[str, str] | None = None
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
