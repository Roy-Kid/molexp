"""Regression: the emergent-plan task-board state layer, end-to-end and offline.

Binding runtime example for spec ``plan-emergent-04-task-board-state`` (ac-008).
A library-external user drives the **public** task-board surface
(``molexp.harness.plan`` + ``molexp.harness.FileArtifactStore``) from an empty
board through pure transitions, full-rewrite persistence, and content-addressed
freezing — importing only documented surfaces. No network, no subprocess, no
server, no CLI; deterministic on every run.

Three ac-008 reference observations, each printed before it is asserted:

1. **pure transitions** — an empty :class:`TaskBoard` (``version == 0``);
   ``place_task`` two tasks (``version`` climbs ``0 -> 1 -> 2``), then
   ``set_task_status`` and ``annotate_feasibility`` push it on (``2 -> 3 -> 4``).
   ``version`` advances strictly monotonically by one per transition, every
   transition returns a NEW board object, and each input board is left
   unchanged field-by-field (immutability law).
2. **full-rewrite persistence** — ``write_board(board_path(run_dir), board)``
   rewrites the whole ``run_dir/plan/task_board.json``; ``read_board(...)``
   reloads the CURRENT state and the reloaded task set equals what was written.
3. **content-addressed freeze** — assembling an :class:`ExperimentPlan`
   (spec + board) and freezing it TWICE through one
   ``FileArtifactStore(root=<tmp>)`` yields EQUAL ``PlanArtifactRef.id`` (and
   ``sha256``) — idempotent content-addressing; changing the board by one task
   and freezing again yields a DIFFERENT id.

Run standalone with ``python regressions/plan-emergent-04-task-board-state.py``;
the process exits 0 and the final line on success is
``plan-emergent-04-task-board-state: ok``. These APIs are synchronous — no
asyncio involved.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from molexp.harness import FileArtifactStore
from molexp.harness.plan import (
    BoardTask,
    ExperimentPlan,
    FeasibilityAnnotation,
    TaskBoard,
    TaskStatus,
    annotate_feasibility,
    board_path,
    freeze_experiment_plan,
    freeze_spec,
    place_task,
    read_board,
    set_task_status,
    write_board,
)

SLUG = "plan-emergent-04-task-board-state"


def _require_task(board: TaskBoard, task_id: str) -> BoardTask:
    """Look up a task that must exist, narrowing away the ``| None``."""
    task = board.task(task_id)
    assert task is not None, f"expected task {task_id!r} on the board"
    return task


def _check_pure_transitions() -> TaskBoard:
    """Observation 1: version advances 0->2->... and inputs never mutate."""
    empty = TaskBoard()
    assert empty.version == 0, f"empty board should start at version 0, got {empty.version}"

    # place two tasks: 0 -> 1 -> 2.
    after_t1 = place_task(empty, BoardTask(id="t1", name="build"))
    after_t2 = place_task(after_t1, BoardTask(id="t2", name="test"))

    # two more transitions on the two-task board: 2 -> 3 -> 4.
    after_status = set_task_status(after_t2, "t1", TaskStatus.BUILDING)
    annotation = FeasibilityAnnotation(reachable=True, rationale="molmcp reachable")
    after_feasibility = annotate_feasibility(after_status, "t2", annotation)

    progression = [
        empty.version,
        after_t1.version,
        after_t2.version,
        after_status.version,
        after_feasibility.version,
    ]
    print(f"[obs-1] board version progression: {progression}")
    assert progression == [0, 1, 2, 3, 4], f"version must climb by one each step, got {progression}"

    # Each transition returns a distinct board object (never the input).
    steps: list[tuple[TaskBoard, TaskBoard]] = [
        (empty, after_t1),
        (after_t1, after_t2),
        (after_t2, after_status),
        (after_status, after_feasibility),
    ]
    for source, produced in steps:
        assert produced is not source, "each transition must return a NEW board object"

    # Inputs are byte-for-byte unchanged after every transition.
    print("[obs-1] input boards unchanged after each transition")
    assert empty.version == 0 and empty.tasks == (), "empty input board mutated"
    assert after_t1.version == 1 and [t.id for t in after_t1.tasks] == ["t1"], (
        "after_t1 input board mutated by place_task"
    )
    assert _require_task(after_t2, "t1").status is TaskStatus.PENDING, (
        "set_task_status mutated its input board's task status"
    )
    assert _require_task(after_status, "t2").feasibility is None, (
        "annotate_feasibility mutated its input board's task"
    )

    # The final board reflects both later transitions.
    assert _require_task(after_feasibility, "t1").status is TaskStatus.BUILDING, (
        "status transition lost"
    )
    assert _require_task(after_feasibility, "t2").feasibility == annotation, (
        "feasibility annotation lost"
    )
    return after_feasibility


def _check_full_rewrite_persistence(run_dir: Path, board: TaskBoard) -> None:
    """Observation 2: write_board full-rewrites; read_board reloads current state."""
    path = board_path(run_dir)
    write_board(path, board)
    reloaded = read_board(path)

    written_ids = sorted(t.id for t in board.tasks)
    reloaded_ids = sorted(t.id for t in reloaded.tasks)
    print(f"[obs-2] wrote task ids {written_ids}; reloaded task ids {reloaded_ids}")
    assert reloaded_ids == written_ids, "reloaded task set must equal what was written"
    assert reloaded == board, "read_board must reload the current persisted board exactly"


def _check_content_addressed_freeze(store: FileArtifactStore, board: TaskBoard) -> None:
    """Observation 3: equal plans freeze to equal ids; a board change flips the id."""
    spec: dict[str, object] = {"title": "task-board demo", "steps": 4}
    plan = ExperimentPlan(spec=spec, board=board)

    first = freeze_experiment_plan(plan, store, created_by="regression")
    second = freeze_experiment_plan(plan, store, created_by="regression")
    print(f"[obs-3] freeze #1 id={first.id}  freeze #2 id={second.id}")
    assert first.id == second.id, "freezing the same plan twice must be idempotent (equal id)"
    assert first.sha256 == second.sha256, "identical plan content must share one sha256"

    changed_board = place_task(board, BoardTask(id="t3", name="report"))
    changed_plan = ExperimentPlan(spec=spec, board=changed_board)
    changed = freeze_experiment_plan(changed_plan, store, created_by="regression")
    print(f"[obs-3] changed-board freeze id={changed.id}")
    assert changed.id != first.id, "a board change must yield a different content-addressed id"

    # A frozen raw spec is likewise content-addressed and independent of the plan id.
    spec_ref = freeze_spec(spec, store, created_by="regression")
    print(f"[obs-3] frozen spec id={spec_ref.id}")
    assert spec_ref.id != first.id, "spec and plan freezes address distinct content"


def main() -> None:
    board = _check_pure_transitions()

    run_dir = Path(tempfile.mkdtemp(prefix="molexp-plan-emergent-04-"))
    try:
        _check_full_rewrite_persistence(run_dir, board)
        _check_content_addressed_freeze(FileArtifactStore(root=run_dir), board)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    print(f"{SLUG}: ok")


if __name__ == "__main__":
    main()
