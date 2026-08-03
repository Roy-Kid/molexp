"""``TaskBoardHandle`` — the immutable task-board contract plan tools read/write.

A ``runtime_checkable`` Protocol structural over any board a plan-tool surface
drives: read methods plus immutable-write methods that each return a NEW board
without mutating the receiver. Production implementation:
:class:`~molexp.harness.plan.disk_board.DiskTaskBoard`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["TaskBoardHandle"]


@runtime_checkable
class TaskBoardHandle(Protocol):
    """Structural contract for an immutable plan-task board."""

    def list_tasks(self) -> list[object]:
        """Return the board's tasks (structured dicts or records)."""
        ...

    def get_task(self, task_id: str) -> object:
        """Return the task with ``task_id``; raise if unknown."""
        ...

    def get_artifact(self, artifact_id: str) -> object:
        """Return the artifact record with ``artifact_id``; raise if unknown."""
        ...

    def place(
        self,
        task_id: str,
        name: str,
        *,
        acceptance: list[str] | tuple[str, ...] | None = None,
    ) -> TaskBoardHandle:
        """Upsert a task onto the board."""
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
