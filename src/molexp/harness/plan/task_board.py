"""Task-board state model + pure transitions for the emergent plan loop.

The :class:`TaskBoard` is the current-state-only record of an experiment's
tasks during the emergent PlanMode loop: a monotone ``version`` counter plus a
tuple of :class:`BoardTask` entries. It carries **no** timestamps and **no**
history — the audit trail lives in the harness event log / artifact lineage,
not in the board itself.

Every mutation is a module-level pure function
(:func:`place_task` / :func:`set_task_status` / :func:`annotate_feasibility` /
:func:`remove_task`) that returns a **new** board with ``version`` incremented
by one and **never** mutates its input. The models are ``frozen=True`` so an
accidental in-place assignment raises instead of silently corrupting shared
state.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from molexp.harness.errors import HarnessError

__all__ = [
    "BoardTask",
    "Difficulty",
    "FeasibilityAnnotation",
    "TaskBoard",
    "TaskNotFoundError",
    "TaskStatus",
    "annotate_feasibility",
    "place_task",
    "remove_task",
    "set_task_status",
]


class TaskStatus(StrEnum):
    """Lifecycle status of a single board task.

    The vocabulary is a linear build ladder plus one off-ramp:

    - ``PENDING`` — placed on the board, not yet started.
    - ``BUILDING`` — production source is being written.
    - ``TESTED`` — its per-task test has run green.
    - ``COMPLETE`` — accepted and done.
    - ``BLOCKED`` — cannot proceed (missing capability, failed feasibility).
    """

    PENDING = "pending"
    BUILDING = "building"
    TESTED = "tested"
    COMPLETE = "complete"
    BLOCKED = "blocked"


class Difficulty(StrEnum):
    """Coarse feasibility difficulty for a probed task.

    ``UNKNOWN`` is the default before any probing has happened; the remaining
    tiers order from cheapest (``TRIVIAL``) to most costly (``HARD``).
    """

    TRIVIAL = "trivial"
    MODERATE = "moderate"
    HARD = "hard"
    UNKNOWN = "unknown"


class FeasibilityAnnotation(BaseModel):
    """Result of probing whether a task is reachable with known capabilities.

    Attributes:
        reachable: Whether the task can be built with the capabilities probed.
        difficulty: Coarse cost tier; ``Difficulty.UNKNOWN`` until assessed.
        rationale: Free-text justification for the reachability verdict.
        probed_refs: Capability / API refs consulted while probing.
    """

    model_config = ConfigDict(frozen=True)

    reachable: bool
    difficulty: Difficulty = Difficulty.UNKNOWN
    rationale: str = ""
    probed_refs: tuple[str, ...] = ()


class BoardTask(BaseModel):
    """One task on the board — identity, acceptance, feasibility, status.

    Attributes:
        id: Stable task identifier (the upsert key on the board).
        name: Human-readable task name.
        acceptance: Acceptance criteria the task must satisfy to be complete.
        feasibility: Latest feasibility probe result, or ``None`` if unprobed.
        status: Current lifecycle status (defaults to ``TaskStatus.PENDING``).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    acceptance: tuple[str, ...] = ()
    feasibility: FeasibilityAnnotation | None = None
    status: TaskStatus = TaskStatus.PENDING


class TaskBoard(BaseModel):
    """Current-state-only board: a version counter plus a tuple of tasks.

    The board holds no timestamps and no history. ``version`` is bumped by
    exactly one on every transition so an optimistic-version writer can detect
    a stale write (see :mod:`molexp.harness.plan.board_store`).

    Attributes:
        version: Monotone revision counter; starts at ``0``.
        tasks: The tasks currently on the board (order preserved).
    """

    model_config = ConfigDict(frozen=True)

    version: int = 0
    tasks: tuple[BoardTask, ...] = ()

    def task(self, task_id: str) -> BoardTask | None:
        """Return the task with ``task_id``, or ``None`` if absent.

        Args:
            task_id: Identifier to look up.

        Returns:
            The matching :class:`BoardTask`, or ``None`` when no task carries
            that id.
        """
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None


class TaskNotFoundError(HarnessError):
    """Raised by a board transition when ``task_id`` is not on the board."""


def _require_index(board: TaskBoard, task_id: str) -> int:
    """Return the index of ``task_id`` on ``board`` or raise.

    Raises:
        TaskNotFoundError: No task on the board carries ``task_id``.
    """
    for index, task in enumerate(board.tasks):
        if task.id == task_id:
            return index
    raise TaskNotFoundError(f"task {task_id!r} not found on board")


def place_task(board: TaskBoard, task: BoardTask) -> TaskBoard:
    """Upsert ``task`` onto ``board``, returning a new bumped-version board.

    If a task with the same ``id`` already exists it is replaced in place
    (no duplication); otherwise ``task`` is appended. The input ``board`` is
    never mutated.

    Args:
        board: The board to derive from.
        task: The task to insert or replace.

    Returns:
        A new :class:`TaskBoard` with ``version`` incremented by one.
    """
    replaced = False
    tasks: list[BoardTask] = []
    for existing in board.tasks:
        if existing.id == task.id:
            tasks.append(task)
            replaced = True
        else:
            tasks.append(existing)
    if not replaced:
        tasks.append(task)
    return board.model_copy(update={"version": board.version + 1, "tasks": tuple(tasks)})


def set_task_status(board: TaskBoard, task_id: str, status: TaskStatus) -> TaskBoard:
    """Set ``task_id``'s status, returning a new bumped-version board.

    Args:
        board: The board to derive from.
        task_id: Identifier of the task to update.
        status: The new lifecycle status.

    Returns:
        A new :class:`TaskBoard` with ``version`` incremented by one.

    Raises:
        TaskNotFoundError: No task on the board carries ``task_id``.
    """
    index = _require_index(board, task_id)
    updated = board.tasks[index].model_copy(update={"status": status})
    tasks = (*board.tasks[:index], updated, *board.tasks[index + 1 :])
    return board.model_copy(update={"version": board.version + 1, "tasks": tasks})


def annotate_feasibility(
    board: TaskBoard, task_id: str, annotation: FeasibilityAnnotation
) -> TaskBoard:
    """Attach a feasibility annotation, returning a new bumped-version board.

    Args:
        board: The board to derive from.
        task_id: Identifier of the task to annotate.
        annotation: The feasibility probe result to attach.

    Returns:
        A new :class:`TaskBoard` with ``version`` incremented by one.

    Raises:
        TaskNotFoundError: No task on the board carries ``task_id``.
    """
    index = _require_index(board, task_id)
    updated = board.tasks[index].model_copy(update={"feasibility": annotation})
    tasks = (*board.tasks[:index], updated, *board.tasks[index + 1 :])
    return board.model_copy(update={"version": board.version + 1, "tasks": tasks})


def remove_task(board: TaskBoard, task_id: str) -> TaskBoard:
    """Drop ``task_id`` from the board, returning a new bumped-version board.

    Args:
        board: The board to derive from.
        task_id: Identifier of the task to remove.

    Returns:
        A new :class:`TaskBoard` with ``version`` incremented by one.

    Raises:
        TaskNotFoundError: No task on the board carries ``task_id``.
    """
    index = _require_index(board, task_id)
    tasks = (*board.tasks[:index], *board.tasks[index + 1 :])
    return board.model_copy(update={"version": board.version + 1, "tasks": tasks})
