"""``TaskBoardHandle`` — the immutable task-board contract plan tools read/write.

A ``runtime_checkable`` Protocol structural over any board a plan-tool surface
drives: three read methods (``list_tasks`` / ``get_task`` / ``get_artifact``) and
four immutable-write methods (``with_task_updated`` / ``mark_complete`` /
``mark_blocked`` / ``apply_patch``) that each return a NEW board without mutating
the receiver. The plan tools never mutate a board in place — they call exactly
one immutable-write and hand back the fresh board, keeping the input byte
identical (immutability law).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["TaskBoardHandle"]


@runtime_checkable
class TaskBoardHandle(Protocol):
    """Structural contract for an immutable plan-task board.

    Read methods return live data; every write method returns a NEW
    ``TaskBoardHandle`` and leaves the receiver unchanged, so a tool that routes
    a mutation through the handle preserves the input board byte-for-byte.
    """

    def list_tasks(self) -> list[object]:
        """Return the board's tasks."""
        ...

    def get_task(self, task_id: str) -> object:
        """Return the task with ``task_id``; raise if it is unknown."""
        ...

    def get_artifact(self, artifact_id: str) -> object:
        """Return the artifact record with ``artifact_id``; raise if unknown."""
        ...

    def with_task_updated(self, task_id: str, changes: dict[str, object]) -> TaskBoardHandle:
        """Return a new board with ``changes`` applied to ``task_id``."""
        ...

    def mark_complete(self, task_id: str) -> TaskBoardHandle:
        """Return a new board with ``task_id`` marked complete."""
        ...

    def mark_blocked(self, task_id: str, reason: str) -> TaskBoardHandle:
        """Return a new board with ``task_id`` marked blocked by ``reason``."""
        ...

    def apply_patch(self, patch: dict[str, object]) -> TaskBoardHandle:
        """Return a new board with ``patch`` applied to the plan."""
        ...
