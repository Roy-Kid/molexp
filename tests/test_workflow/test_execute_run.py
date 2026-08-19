"""One-step tracked execution — ``execute_run`` / ``Run.execute`` (runset-api sub-task 1).

``molexp.workflow.execute_run(workflow, run)`` folds the seven-touchpoint
driver dance (``run.start()`` + ``WorkflowRuntime().execute(run_context=...)``
+ asyncio plumbing) into one call that reuses the exact same execution path
as ``molexp run``: RunContext lifecycle (status machine, ``ops`` sidecar,
heartbeat) + the workflow engine. ``Run.execute`` is the workspace-side sugar
reached through the ``set_run_executor`` inversion seam (workspace never
imports workflow).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import molexp as me
from molexp.workflow import (
    RunFailedError,
    RunNotExecutableError,
    WorkflowCompiler,
    aexecute_run,
    execute_run,
)
from molexp.workspace.models import RunStatus


def _make_run(tmp_path: Path, params: dict | None = None):
    ws = me.Workspace(tmp_path / "ws", name="lab")
    exp = ws.add_project("demo").add_experiment("pipeline")
    return exp.add_run(params=params if params is not None else {"x": 3})


def _build_wf() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="pipeline")

    @wf.task
    def double(x: int) -> int:
        return x * 2

    @wf.task(depends_on=["double"])
    def summarize(double: int) -> str:
        return f"got {double}"

    return wf


class TestExecuteRun:
    def test_one_step_execute_returns_per_task_outputs(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        result = execute_run(_build_wf().compile(), run)
        assert result.status == "succeeded"
        assert result.outputs["double"] == 6
        assert result.outputs["summarize"] == "got 6"
        assert run.status == RunStatus.SUCCEEDED.value
        assert len(run.execution_history) == 1

    def test_task_failure_raises_and_persists_failed_run(self, tmp_path: Path) -> None:
        wf = WorkflowCompiler(name="boom")

        @wf.task
        def explode(x: int) -> int:
            raise ZeroDivisionError("bad cell")

        run = _make_run(tmp_path)
        with pytest.raises(RunFailedError) as excinfo:
            execute_run(wf, run)
        assert run.status == RunStatus.FAILED.value
        # The exception surfaces WHY and carries the partial WorkflowResult.
        assert "ZeroDivisionError" in str(excinfo.value)
        assert excinfo.value.result.status == "failed"

    def test_task_failure_persists_real_traceback_in_error_txt(self, tmp_path: Path) -> None:
        """The engine swallows the task exception, but its live traceback must
        still reach ``executions/<exec_id>/error.txt`` — the documented "with
        traceback" trace file, not a placeholder note."""
        wf = WorkflowCompiler(name="boom")

        @wf.task
        def explode(x: int) -> int:
            raise ZeroDivisionError("bad cell")

        run = _make_run(tmp_path)
        with pytest.raises(RunFailedError):
            execute_run(wf, run)

        exec_id = run.execution_history[-1].execution_id
        error_txt = Path(str(run.run_dir)) / "executions" / exec_id / "error.txt"
        assert error_txt.exists()
        content = error_txt.read_text()
        assert "Traceback (most recent call last):" in content
        assert "ZeroDivisionError: bad cell" in content
        # The task-body frame is part of the stack — this is a REAL traceback.
        assert 'raise ZeroDivisionError("bad cell")' in content
        assert "No Python traceback was captured" not in content

    def test_succeeded_refuses_even_with_rerun(self, tmp_path: Path) -> None:
        """A succeeded run is refused with or without ``rerun`` — the succeeded
        check precedes the verb domain, so ``rerun=True`` cannot resurrect a
        done run (verb domain is failed/cancelled only)."""
        run = _make_run(tmp_path)
        execute_run(_build_wf(), run)
        with pytest.raises(RunNotExecutableError, match="succeeded"):
            execute_run(_build_wf(), run, rerun=True)
        assert len(run.execution_history) == 1

    def test_fresh_requires_rerun(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        with pytest.raises(ValueError, match="rerun=True"):
            execute_run(_build_wf(), run, fresh=True)

    def test_failed_then_default_resumes_same_execution(self, tmp_path: Path) -> None:
        """Default retry of a failed run is *resume*: reopen the same
        execution, seed completed nodes, recompute only the rest."""
        flag = tmp_path / "healed"
        first_calls: list[int] = []

        def build() -> WorkflowCompiler:
            wf = WorkflowCompiler(name="healing")

            @wf.task
            def stage_a(x: int) -> int:
                first_calls.append(x)
                return x + 1

            @wf.task(depends_on=["stage_a"])
            def stage_b(stage_a: int) -> int:
                if not Path(str(flag)).exists():
                    raise RuntimeError("not healed yet")
                return stage_a * 100

            return wf

        run = _make_run(tmp_path, params={"x": 1})
        with pytest.raises(RunFailedError):
            execute_run(build(), run)
        assert run.status == RunStatus.FAILED.value
        exec_ids_before = [r.execution_id for r in run.execution_history]

        flag.write_text("ok")
        result = execute_run(build(), run)
        assert result.status == "succeeded"
        assert result.outputs["stage_b"] == 200
        # Same execution reopened — no new attempt appended.
        assert [r.execution_id for r in run.execution_history] == exec_ids_before
        # stage_a was seeded from the persisted node output, not recomputed.
        assert first_calls == [1]

    def test_failed_then_rerun_opens_new_execution(self, tmp_path: Path) -> None:
        wf_fail = WorkflowCompiler(name="always-fails")

        @wf_fail.task
        def explode(x: int) -> int:
            raise RuntimeError("boom")

        run = _make_run(tmp_path)
        with pytest.raises(RunFailedError):
            execute_run(wf_fail, run)
        result = execute_run(_build_wf(), run, rerun=True)
        assert result.status == "succeeded"
        assert len(run.execution_history) == 2

    def test_running_run_raises(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        run.materialize()
        run.update_ops(lambda state: state.model_copy(update={"status": RunStatus.RUNNING}))
        with pytest.raises(RunNotExecutableError, match="cancel"):
            execute_run(_build_wf(), run)

    def test_sync_facade_inside_event_loop_raises(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)

        async def inner() -> None:
            execute_run(_build_wf(), run)

        with pytest.raises(RuntimeError, match="aexecute"):
            asyncio.run(inner())

    def test_async_variant(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        result = asyncio.run(aexecute_run(_build_wf(), run))
        assert result.status == "succeeded"
        assert result.outputs["summarize"] == "got 6"


class TestRunExecuteMethod:
    def test_run_execute_delegates_through_seam(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path, params={"x": 4})
        result = run.execute(_build_wf().compile())
        assert result.status == "succeeded"
        assert result.outputs["summarize"] == "got 8"
        assert run.status == RunStatus.SUCCEEDED.value
