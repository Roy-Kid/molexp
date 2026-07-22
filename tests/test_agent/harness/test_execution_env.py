"""Tests for ``molexp.agent.execution_env`` — real-subprocess ``LocalExecutionEnv``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from molexp.agent.execution_env import (
    ExecResult,
    ExecutionError,
    LocalExecutionEnv,
)


class TestLocalExecutionEnv:
    def test_successful_command_captures_stdout_and_zero_exit(self, tmp_path: Path) -> None:
        env = LocalExecutionEnv(scratch_dir=tmp_path)
        result = env.exec([sys.executable, "-c", "print('hello-harness')"])
        assert isinstance(result, ExecResult)
        assert result.exit_code == 0
        assert "hello-harness" in result.stdout

    def test_nonzero_exit_is_a_result_not_an_error(self, tmp_path: Path) -> None:
        """A non-zero exit yields an ``ExecResult`` with captured stderr, never a raise."""
        env = LocalExecutionEnv(scratch_dir=tmp_path)
        result = env.exec(
            [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"]
        )
        assert result.exit_code == 3
        assert "boom" in result.stderr

    def test_cwd_confines_the_child_working_directory(self, tmp_path: Path) -> None:
        workdir = tmp_path / "confined"
        workdir.mkdir()
        env = LocalExecutionEnv(scratch_dir=tmp_path)
        result = env.exec(
            [sys.executable, "-c", "import os; print(os.getcwd())"],
            cwd=workdir,
        )
        assert result.exit_code == 0
        assert str(workdir.resolve()) in result.stdout

    def test_spawn_failure_raises_execution_error(self, tmp_path: Path) -> None:
        env = LocalExecutionEnv(scratch_dir=tmp_path)
        with pytest.raises(ExecutionError):
            env.exec(["this-binary-does-not-exist-xyz"])
