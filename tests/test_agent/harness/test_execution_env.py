"""``ExecutionEnv`` tests — real subprocess + fake parity (spec ac-007)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from molexp.agent.harness.execution_env import (
    ExecResult,
    ExecutionEnv,
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


def test_local_env_timeout_raises_execution_error(tmp_path: Path) -> None:
    env = LocalExecutionEnv(scratch_dir=tmp_path)
    with pytest.raises(ExecutionError) as excinfo:
        env.exec(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout=0.5,
        )
    assert "timeout" in str(excinfo.value).lower()


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


def test_local_env_defaults_cwd_to_scratch_dir(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    env = LocalExecutionEnv(scratch_dir=scratch)
    assert env.scratch_dir == scratch.resolve()
    result = env.exec([sys.executable, "-c", "import os; print(os.getcwd())"])
    assert str(scratch.resolve()) in result.stdout


def test_local_env_scratch_dir_is_created(tmp_path: Path) -> None:
    scratch = tmp_path / "made-on-demand"
    assert not scratch.exists()
    env = LocalExecutionEnv(scratch_dir=scratch)
    _ = env.scratch_dir
    assert scratch.exists()


def test_local_env_spawn_failure_raises_execution_error(tmp_path: Path) -> None:
    env = LocalExecutionEnv(scratch_dir=tmp_path)
    with pytest.raises(ExecutionError):
        env.exec(["this-binary-does-not-exist-xyz"])


def test_local_env_passes_env_vars(tmp_path: Path) -> None:
    env = LocalExecutionEnv(scratch_dir=tmp_path)
    result = env.exec(
        [sys.executable, "-c", "import os; print(os.environ.get('MOLEXP_T', 'unset'))"],
        env={"MOLEXP_T": "wired"},
    )
    assert "wired" in result.stdout


# ── Protocol conformance ───────────────────────────────────────────────────


def test_local_env_satisfies_protocol(tmp_path: Path) -> None:
    env = LocalExecutionEnv(scratch_dir=tmp_path)
    assert isinstance(env, ExecutionEnv)


def test_exec_result_is_frozen() -> None:
    from pydantic import ValidationError

    result = ExecResult(stdout="", stderr="", exit_code=0)
    with pytest.raises(ValidationError):
        result.exit_code = 1  # type: ignore[misc]
