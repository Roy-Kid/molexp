"""RED tests for ``CommandTask`` / ``CommandError`` (spec
framework-scaffolding-parity-04-command-task).

``CommandTask`` collapses the repeated "run a command, raise on non-zero
exit" block into one reusable workflow task. It accepts exactly one of:

* ``argv`` — a ``list[str]`` run via ``subprocess.run`` (capturing
  stdout/stderr), or
* ``runner`` — a zero-arg callable returning an object exposing
  ``.returncode`` / ``.stdout`` / ``.stderr`` (the "molpy-style" path).

On a zero exit ``execute`` returns the normalized result; on a non-zero
exit it raises :class:`CommandError` (a :class:`WorkflowError` subclass)
carrying the command's stderr/stdout.

Happy-path tests drive a real 1-task workflow so ``execute`` receives a
genuine :class:`TaskContext`. Failure / validation / edge tests call
``execute`` directly with a minimal ctx stub because the runtime collapses
any raised exception into ``result.status == "failed"`` and would hide the
``CommandError`` we need to assert on; the argv/runner paths do not read
``ctx`` so a stub suffices.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from molexp.workflow import (
    CommandError,
    CommandTask,
    WorkflowCompiler,
    WorkflowError,
    WorkflowRuntime,
)

# ── helpers ──────────────────────────────────────────────────────────────────


class _CompletedStub:
    """Duck-typed stand-in for ``subprocess.CompletedProcess`` returned by a
    ``runner`` callable: exposes ``.returncode`` / ``.stdout`` / ``.stderr``."""

    def __init__(self, *, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _CtxStub:
    """Lightweight ctx for direct ``execute`` calls. The argv/runner paths
    ignore ``ctx`` entirely; this only needs to be a non-``None`` object."""

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


# ── ac-006: public exports ─────────────────────────────────────────────────────


class TestExports:
    def test_command_task_and_error_in_all(self) -> None:
        import molexp.workflow as wf_pkg

        assert "CommandTask" in wf_pkg.__all__
        assert "CommandError" in wf_pkg.__all__

    def test_command_error_also_importable_from_types(self) -> None:
        from molexp.workflow.types import CommandError as TypesCommandError

        assert TypesCommandError is CommandError

    def test_command_error_is_workflow_error_subclass(self) -> None:
        assert issubclass(CommandError, WorkflowError)


# ── ac-004: dual construction + fail-fast validation ──────────────────────────


class TestConstruction:
    def test_argv_positional_constructs(self) -> None:
        task = CommandTask(["true"])
        assert isinstance(task, CommandTask)

    def test_argv_keyword_constructs(self) -> None:
        task = CommandTask(argv=["true"])
        assert isinstance(task, CommandTask)

    def test_runner_keyword_constructs(self) -> None:
        task = CommandTask(runner=lambda: _CompletedStub(returncode=0, stdout="", stderr=""))
        assert isinstance(task, CommandTask)

    def test_both_argv_and_runner_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            CommandTask(["x"], runner=lambda: _CompletedStub(returncode=0, stdout="", stderr=""))

    def test_neither_argv_nor_runner_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            CommandTask()


# ── ac-003: success returns normalized result (both paths) ────────────────────


@pytest.mark.asyncio
class TestSuccess:
    async def test_argv_success_via_workflow_returns_zero_returncode(self) -> None:
        spec = (
            WorkflowCompiler(name="argv-ok")
            .add(CommandTask(_normal_exit_argv("ok")), name="cmd")
            .compile()
        )
        result = await WorkflowRuntime().execute(spec)
        assert result.status == "completed"
        out = result.outputs["cmd"]
        assert out.returncode == 0
        assert out.stdout == "ok"

    async def test_argv_success_direct_execute_returns_result(self) -> None:
        out = await CommandTask(_normal_exit_argv("hey")).execute(_CtxStub())
        assert out.returncode == 0
        assert out.stdout == "hey"

    async def test_runner_success_returns_normalized_result(self) -> None:
        task = CommandTask(runner=lambda: _CompletedStub(returncode=0, stdout="out", stderr=""))
        out = await task.execute(_CtxStub())
        assert out.returncode == 0
        assert out.stdout == "out"
        assert out.stderr == ""

    async def test_runner_success_via_workflow(self) -> None:
        spec = (
            WorkflowCompiler(name="runner-ok")
            .add(
                CommandTask(
                    runner=lambda: _CompletedStub(returncode=0, stdout="produced", stderr="")
                ),
                name="cmd",
            )
            .compile()
        )
        result = await WorkflowRuntime().execute(spec)
        assert result.status == "completed"
        assert result.outputs["cmd"].returncode == 0
        assert result.outputs["cmd"].stdout == "produced"


# ── ac-001 / ac-002: non-zero exit raises CommandError carrying stderr/stdout ──


@pytest.mark.asyncio
class TestFailure:
    async def test_argv_nonzero_raises_command_error(self) -> None:
        task = CommandTask(_nonzero_argv(stderr="boom", code=3))
        with pytest.raises(CommandError) as excinfo:
            await task.execute(_CtxStub())
        err = excinfo.value
        assert isinstance(err, WorkflowError)
        assert err.returncode == 3
        assert "boom" in str(err)
        assert "boom" in err.stderr

    async def test_runner_nonzero_raises_command_error_with_streams(self) -> None:
        task = CommandTask(
            runner=lambda: _CompletedStub(returncode=2, stdout="some-out", stderr="bad-thing")
        )
        with pytest.raises(CommandError) as excinfo:
            await task.execute(_CtxStub())
        err = excinfo.value
        assert isinstance(err, WorkflowError)
        assert err.returncode == 2
        assert err.stderr == "bad-thing"
        assert err.stdout == "some-out"
        assert "bad-thing" in str(err)


# ── edge cases ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestEdgeCases:
    async def test_nonzero_with_empty_stderr_surfaces_stdout_in_message(self) -> None:
        task = CommandTask(
            runner=lambda: _CompletedStub(returncode=1, stdout="fallback-detail", stderr="")
        )
        with pytest.raises(CommandError) as excinfo:
            await task.execute(_CtxStub())
        assert "fallback-detail" in str(excinfo.value)

    async def test_zero_exit_with_nonempty_stderr_does_not_raise(self) -> None:
        task = CommandTask(
            runner=lambda: _CompletedStub(returncode=0, stdout="", stderr="just a warning")
        )
        out = await task.execute(_CtxStub())
        assert out.returncode == 0
        assert out.stderr == "just a warning"


# ── ac-005: raise-on-nonzero lives once in CommandTask, not per wrapper ────────


class _WrapperA(CommandTask):
    """Wrapper task delegating to CommandTask — adds NO raise-on-nonzero block."""


class _WrapperB(CommandTask):
    """A second wrapper, also pure delegation."""


@pytest.mark.asyncio
class TestNoDuplicationReuse:
    async def test_both_wrappers_delegate_raise_to_command_task(self) -> None:
        for wrapper_cls, code in ((_WrapperA, 4), (_WrapperB, 5)):
            wrapper = wrapper_cls(_nonzero_argv(stderr="from-wrapper", code=code))
            with pytest.raises(CommandError) as excinfo:
                await wrapper.execute(_CtxStub())
            assert excinfo.value.returncode == code

    def test_wrapper_sources_contain_no_own_raise_block(self) -> None:
        import inspect
        import re

        for wrapper_cls in (_WrapperA, _WrapperB):
            src = inspect.getsource(wrapper_cls)
            assert "returncode" not in src
            assert not re.search(r"raise\s+RuntimeError", src)
