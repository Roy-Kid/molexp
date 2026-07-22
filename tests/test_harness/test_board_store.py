"""RED tests for ``molexp.harness.plan.board_store`` — persistence + versioning.

Pins the FULL-rewrite, current-state-only persistence of a ``TaskBoard``
under ``run_dir/plan/task_board.json`` and its optimistic-version guard:
a stale ``expected_version`` raises ``BoardVersionConflict``; ``None`` is
last-write-wins. Reading a missing file yields an empty ``TaskBoard()``.

Function-level unit tests only (real ``tmp_path``, no subprocess). The
production subpackage ``molexp.harness.plan`` does not exist yet: a
``ModuleNotFoundError`` at import time is the valid RED signal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.harness.plan import (
    BoardTask,
    BoardVersionConflict,
    TaskBoard,
    board_path,
    read_board,
    write_board,
)


class TestBoardPath:
    def test_board_path_layout(self, tmp_path: Path) -> None:
        assert board_path(tmp_path) == tmp_path / "plan" / "task_board.json"


class TestRoundtrip:
    def test_write_then_read_returns_equivalent_board(self, tmp_path: Path) -> None:
        path = board_path(tmp_path)
        board = TaskBoard(version=2, tasks=(BoardTask(id="t1", name="build"),))
        write_board(path, board)
        assert read_board(path) == board

    def test_read_missing_file_returns_empty_board(self, tmp_path: Path) -> None:
        assert read_board(board_path(tmp_path)) == TaskBoard()


class TestFullRewrite:
    def test_second_write_replaces_previous_state_entirely(self, tmp_path: Path) -> None:
        path = board_path(tmp_path)
        first = TaskBoard(
            version=1,
            tasks=(BoardTask(id="t1", name="a"), BoardTask(id="t2", name="b")),
        )
        write_board(path, first)
        second = TaskBoard(version=2, tasks=(BoardTask(id="t3", name="c"),))
        write_board(path, second)

        loaded = read_board(path)
        assert loaded == second
        # Previous tasks must NOT be residual (full rewrite, current state only).
        assert [t.id for t in loaded.tasks] == ["t3"]


class TestOptimisticVersion:
    def test_stale_expected_version_raises_conflict(self, tmp_path: Path) -> None:
        path = board_path(tmp_path)
        write_board(path, TaskBoard(version=5))  # on-disk version advances to 5
        with pytest.raises(BoardVersionConflict):
            write_board(path, TaskBoard(version=6), expected_version=3)

    def test_matching_expected_version_succeeds(self, tmp_path: Path) -> None:
        path = board_path(tmp_path)
        write_board(path, TaskBoard(version=5))
        write_board(path, TaskBoard(version=6), expected_version=5)
        assert read_board(path).version == 6

    def test_none_expected_version_is_last_write_wins(self, tmp_path: Path) -> None:
        path = board_path(tmp_path)
        write_board(path, TaskBoard(version=5))
        write_board(path, TaskBoard(version=2), expected_version=None)
        assert read_board(path).version == 2
