"""Tests for :class:`molexp.workflow.command_task.CommandTask`.

``CommandTask`` collapses the recurring "run a command, raise on non-zero
exit" block into one reusable workflow task. It accepts exactly one of:

* ``argv`` — a ``list[str]`` run via ``subprocess.run`` (capturing streams), or
* ``runner`` — a zero-arg callable returning an object exposing
  ``.returncode`` / ``.stdout`` / ``.stderr`` (the "molpy-style" path).

On a zero exit ``execute`` returns the normalized :class:`CommandResult`; on a
non-zero exit it raises :class:`CommandError` (a :class:`WorkflowError`)
carrying the command's streams. The argv/runner paths ignore ``ctx``, so
``execute`` is driven directly with a minimal ctx stub.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from molexp.workflow import CommandError, CommandTask, WorkflowError


class _CompletedStub:
    """Duck-typed stand-in for ``subprocess.CompletedProcess`` returned by a
    ``runner`` callable: exposes ``.returncode`` / ``.stdout`` / ``.stderr``."""

    def __init__(self, *, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _CtxStub:
    """Non-``None`` ctx for direct ``execute`` calls; argv/runner ignore it."""

    run_context: Any = None
    inputs: Any = None
    config: Any = None


def _normal_exit_argv(stdout: str = "hi") -> list[str]:
    return [sys.executable, "-c", f"import sys; sys.stdout.write({stdout!r})"]


def _nonzero_argv(*, stderr: str = "", stdout: str = "", code: int = 3) -> list[str]:
    body = (
        f"import sys; sys.stdout.write({stdout!r}); sys.stderr.write({stderr!r}); sys.exit({code})"
    )
    return [sys.executable, "-c", body]


class TestCommandTask:
    # ── dual-construction fail-fast validation (both boundaries) ──────────────

    def test_both_argv_and_runner_rejected(self) -> None:
        with pytest.raises(ValueError):
            CommandTask(["x"], runner=lambda: _CompletedStub(returncode=0, stdout="", stderr=""))

    def test_neither_argv_nor_runner_rejected(self) -> None:
        with pytest.raises(ValueError):
            CommandTask()

    # ── zero exit → normalized result (both paths) ────────────────────────────

    @pytest.mark.asyncio
    async def test_argv_zero_exit_returns_normalized_result(self) -> None:
        out = await CommandTask(_normal_exit_argv("ok")).execute(_CtxStub())
        assert out.returncode == 0
        assert out.stdout == "ok"

    @pytest.mark.asyncio
    async def test_runner_zero_exit_returns_normalized_result(self) -> None:
        task = CommandTask(runner=lambda: _CompletedStub(returncode=0, stdout="out", stderr=""))
        out = await task.execute(_CtxStub())
        assert (out.returncode, out.stdout, out.stderr) == (0, "out", "")

    # ── non-zero exit → CommandError carrying the streams (both paths) ────────

    @pytest.mark.asyncio
    async def test_argv_non_zero_exit_raises_command_error(self) -> None:
        with pytest.raises(CommandError) as excinfo:
            await CommandTask(_nonzero_argv(stderr="boom", code=3)).execute(_CtxStub())
        err = excinfo.value
        assert isinstance(err, WorkflowError)
        assert err.returncode == 3
        assert "boom" in str(err)
        assert "boom" in err.stderr

    @pytest.mark.asyncio
    async def test_runner_non_zero_exit_raises_command_error_with_streams(self) -> None:
        task = CommandTask(
            runner=lambda: _CompletedStub(returncode=2, stdout="some-out", stderr="bad-thing")
        )
        with pytest.raises(CommandError) as excinfo:
            await task.execute(_CtxStub())
        err = excinfo.value
        assert (err.returncode, err.stdout, err.stderr) == (2, "some-out", "bad-thing")
        assert "bad-thing" in str(err)

    # ── return code is the sole trigger; message falls back to stdout ─────────

    @pytest.mark.asyncio
    async def test_zero_exit_with_stderr_does_not_raise(self) -> None:
        task = CommandTask(
            runner=lambda: _CompletedStub(returncode=0, stdout="", stderr="just a warning")
        )
        out = await task.execute(_CtxStub())
        assert out.returncode == 0
        assert out.stderr == "just a warning"

    @pytest.mark.asyncio
    async def test_empty_stderr_surfaces_stdout_in_error_message(self) -> None:
        task = CommandTask(
            runner=lambda: _CompletedStub(returncode=1, stdout="fallback-detail", stderr="")
        )
        with pytest.raises(CommandError) as excinfo:
            await task.execute(_CtxStub())
        assert "fallback-detail" in str(excinfo.value)
