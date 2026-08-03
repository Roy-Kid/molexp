"""Disk-backed :class:`TaskBoardHandle` — production board the plan tools drive.

Reads and full-rewrites ``plan/task_board.json`` through
:func:`~molexp.harness.plan.board_store.read_board` /
:func:`~molexp.harness.plan.board_store.write_board`. Every write method
returns a **new** :class:`DiskTaskBoard` instance (same path) so the
immutable-handle contract holds: callers never share a stale in-memory
snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from molexp.harness.plan.board_store import read_board, write_board
from molexp.harness.plan.task_board import (
    BoardTask,
    TaskBoard,
    TaskNotFoundError,
    TaskStatus,
    place_task,
    remove_task,
    set_task_status,
)

if TYPE_CHECKING:
    from molexp.harness.store.artifact_store import ArtifactStore

__all__ = ["DiskTaskBoard"]


def _task_public(task: BoardTask) -> dict[str, Any]:
    """JSON-friendly view of one board task for tool results."""
    return {
        "id": task.id,
        "name": task.name,
        "status": str(task.status),
        "acceptance": list(task.acceptance),
        "feasibility": (
            None
            if task.feasibility is None
            else {
                "reachable": task.feasibility.reachable,
                "difficulty": str(task.feasibility.difficulty),
                "rationale": task.feasibility.rationale,
                "probed_refs": list(task.feasibility.probed_refs),
            }
        ),
    }


def _task_update_fields(changes: Mapping[Any, object]) -> dict[str, Any]:
    """Map a patch's ``{name,acceptance,status}`` changes to model_copy fields.

    Shared by :meth:`DiskTaskBoard.with_task_updated` and the ``update`` half of
    :meth:`DiskTaskBoard.apply_patch` — the same three-key vocabulary, written
    once. Unknown keys are ignored (the board owns its own schema); the key type
    is ``Any`` because one caller hands over an isinstance-narrowed JSON dict.
    """
    from molexp.harness.plan.task_board import coerce_acceptance

    update: dict[str, Any] = {}
    for key, value in changes.items():
        if key == "name":
            update["name"] = str(value)
        elif key == "acceptance":
            update["acceptance"] = coerce_acceptance(value)
        elif key == "status":
            update["status"] = TaskStatus(str(value))
    return update


class DiskTaskBoard:
    """Production :class:`~molexp.harness.plan_tools.TaskBoardHandle` on disk.

    Args:
        path: Absolute path to ``task_board.json`` (see
            :func:`~molexp.harness.plan.board_store.board_path`).
        artifact_store: Optional store for :meth:`get_artifact`; when
            ``None``, artifact lookups raise :class:`KeyError`.
    """

    def __init__(
        self,
        path: Path,
        *,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self._path = Path(path)
        self._artifact_store = artifact_store

    @property
    def path(self) -> Path:
        """On-disk board path."""
        return self._path

    def snapshot(self) -> TaskBoard:
        """Load the current board (empty board if the file is absent)."""
        return read_board(self._path)

    def list_tasks(self) -> list[object]:
        return [_task_public(t) for t in self.snapshot().tasks]

    def get_task(self, task_id: str) -> object:
        task = self.snapshot().task(task_id)
        if task is None:
            raise TaskNotFoundError(f"task {task_id!r} not found on board")
        return _task_public(task)

    def get_artifact(self, artifact_id: str) -> object:
        if self._artifact_store is None:
            raise KeyError(artifact_id)
        raw = self._artifact_store.get(artifact_id)
        return {"artifact_id": artifact_id, "bytes": len(raw)}

    def place(
        self,
        task_id: str,
        name: str,
        *,
        acceptance: list[str] | tuple[str, ...] | None = None,
    ) -> DiskTaskBoard:
        """Upsert a task onto the board and persist."""
        from molexp.harness.plan.task_board import coerce_acceptance

        acc = coerce_acceptance(acceptance)
        board = place_task(
            self.snapshot(),
            BoardTask(id=task_id, name=name, acceptance=acc),
        )
        write_board(self._path, board)
        return DiskTaskBoard(self._path, artifact_store=self._artifact_store)

    def with_task_updated(self, task_id: str, changes: dict[str, object]) -> DiskTaskBoard:
        board = self.snapshot()
        task = board.task(task_id)
        if task is None:
            raise TaskNotFoundError(f"task {task_id!r} not found on board")
        updated = task.model_copy(update=_task_update_fields(changes))
        write_board(self._path, place_task(board, updated))
        return DiskTaskBoard(self._path, artifact_store=self._artifact_store)

    def mark_complete(self, task_id: str) -> DiskTaskBoard:
        write_board(self._path, set_task_status(self.snapshot(), task_id, TaskStatus.COMPLETE))
        return DiskTaskBoard(self._path, artifact_store=self._artifact_store)

    def mark_blocked(self, task_id: str, reason: str) -> DiskTaskBoard:
        board = self.snapshot()
        task = board.task(task_id)
        if task is None:
            raise TaskNotFoundError(f"task {task_id!r} not found on board")
        # Board model has no block_reason field yet — append reason to acceptance
        # so the operator can still see why it blocked.
        note = f"blocked: {reason}" if reason else "blocked"
        acceptance = (*task.acceptance, note) if note not in task.acceptance else task.acceptance
        updated = task.model_copy(update={"status": TaskStatus.BLOCKED, "acceptance": acceptance})
        write_board(self._path, place_task(board, updated))
        return DiskTaskBoard(self._path, artifact_store=self._artifact_store)

    def apply_patch(self, patch: dict[str, object]) -> DiskTaskBoard:
        """Apply a plan-level patch: place/update/remove tasks.

        Recognised keys (all optional):

        * ``tasks`` — list of ``{id, name, acceptance?}`` to upsert
        * ``update`` — map ``task_id → changes`` (same as :meth:`with_task_updated`)
        * ``remove`` — list of task ids to drop
        """
        board = self.snapshot()
        tasks_raw = patch.get("tasks")
        if isinstance(tasks_raw, list):
            for item in tasks_raw:
                if not isinstance(item, dict):
                    continue
                tid = str(item.get("id") or "").strip()
                name = str(item.get("name") or tid).strip()
                if not tid:
                    continue
                from molexp.harness.plan.task_board import coerce_acceptance

                acc = coerce_acceptance(item.get("acceptance"))
                board = place_task(board, BoardTask(id=tid, name=name, acceptance=acc))

        updates = patch.get("update")
        if isinstance(updates, dict):
            for tid, changes in updates.items():
                if not isinstance(changes, dict):
                    continue
                task = board.task(str(tid))
                if task is None:
                    continue
                board = place_task(board, task.model_copy(update=_task_update_fields(changes)))

        remove_raw = patch.get("remove")
        if isinstance(remove_raw, list):
            for tid in remove_raw:
                try:
                    board = remove_task(board, str(tid))
                except TaskNotFoundError:
                    continue

        write_board(self._path, board)
        return DiskTaskBoard(self._path, artifact_store=self._artifact_store)
