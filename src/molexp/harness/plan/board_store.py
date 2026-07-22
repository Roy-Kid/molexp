"""Full-rewrite, current-state-only persistence of a :class:`TaskBoard`.

A board lives at ``run_dir/plan/task_board.json`` and is always written as a
complete snapshot — never patched — so the on-disk file is the whole current
state with no residual tasks from a prior write.

:func:`write_board` supports two concurrency policies:

- **Optimistic version** (``expected_version`` given): the on-disk board's
  ``version`` must equal ``expected_version`` or the write is rejected with
  :class:`BoardVersionConflict`, so a stale writer never clobbers a newer
  board.
- **Last-write-wins** (``expected_version=None``): the write always lands,
  overwriting whatever is on disk.
"""

from __future__ import annotations

from pathlib import Path

from molexp.atomicio import atomic_write_json
from molexp.harness.errors import HarnessError
from molexp.harness.plan.task_board import TaskBoard

__all__ = [
    "BoardVersionConflict",
    "board_path",
    "read_board",
    "write_board",
]


class BoardVersionConflict(HarnessError):
    """Raised when an optimistic-version write hits a stale ``expected_version``.

    Signals that the on-disk board advanced past the version the writer read,
    so the write would clobber a newer state. The caller should re-read and
    retry.
    """


def board_path(run_dir: Path) -> Path:
    """Return the canonical task-board path under a run directory.

    Args:
        run_dir: The run directory root.

    Returns:
        ``run_dir/plan/task_board.json``.
    """
    return run_dir / "plan" / "task_board.json"


def read_board(path: Path) -> TaskBoard:
    """Load a board from ``path``, or an empty board if the file is absent.

    Args:
        path: The task-board JSON path (see :func:`board_path`).

    Returns:
        The persisted :class:`TaskBoard`, or a fresh ``TaskBoard()`` when the
        file does not exist.
    """
    if not path.exists():
        return TaskBoard()
    return TaskBoard.model_validate_json(path.read_text(encoding="utf-8"))


def write_board(
    path: Path,
    board: TaskBoard,
    *,
    expected_version: int | None = None,
) -> None:
    """Persist ``board`` to ``path`` as a full-state rewrite.

    When ``expected_version`` is supplied, the current on-disk board's
    ``version`` is read first and must match it (optimistic concurrency);
    otherwise the write is last-write-wins. The write itself is atomic and
    replaces the file entirely, so no prior task survives.

    Args:
        path: The task-board JSON path (see :func:`board_path`).
        board: The board snapshot to persist.
        expected_version: If not ``None``, the version the on-disk board must
            currently have; a mismatch raises :class:`BoardVersionConflict`.

    Raises:
        BoardVersionConflict: ``expected_version`` was given and the on-disk
            board's version differs from it.
    """
    if expected_version is not None:
        current = read_board(path).version
        if current != expected_version:
            raise BoardVersionConflict(
                f"board at {path} is version {current}, expected {expected_version}"
            )
    atomic_write_json(path, board.model_dump(mode="json"))
