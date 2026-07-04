"""``ExecutionEnv`` tests — real subprocess (spec ac-007)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from molexp.agent.execution_env import (
    ExecResult,
    ExecutionError,
    LocalExecutionEnv,
)

# ── LocalExecutionEnv real subprocess ──────────────────────────────────────


def test_local_env_runs_a_normal_command(tmp_path: Path) -> None:
    env = LocalExecutionEnv(scratch_dir=tmp_path)
    result = env.exec([sys.executable, "-c", "print('hello-harness')"])
    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert "hello-harness" in result.stdout


def test_local_env_captures_nonzero_exit_and_stderr(tmp_path: Path) -> None:
    env = LocalExecutionEnv(scratch_dir=tmp_path)
    result = env.exec([sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"])
    assert result.exit_code == 3
    assert "boom" in result.stderr


def test_local_env_runs_in_confined_cwd(tmp_path: Path) -> None:
    workdir = tmp_path / "confined"
    workdir.mkdir()
    env = LocalExecutionEnv(scratch_dir=tmp_path)
    result = env.exec(
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        cwd=workdir,
    )
    assert result.exit_code == 0
    assert str(workdir.resolve()) in result.stdout


def test_local_env_spawn_failure_raises_execution_error(tmp_path: Path) -> None:
    env = LocalExecutionEnv(scratch_dir=tmp_path)
    with pytest.raises(ExecutionError):
        env.exec(["this-binary-does-not-exist-xyz"])
