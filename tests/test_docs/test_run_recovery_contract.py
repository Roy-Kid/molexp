"""Pin the run-recovery contracts the docs promise.

``docs/guide/workflow-persistence.md`` and ``docs/guide/workspace-api.md``
promise that a failed attempt leaves ``executions/<exec_id>/error.txt``
explaining WHY, and that ``molexp run`` exposes the ``--resume`` /
``--rerun`` / ``--fresh`` verbs. These tests keep those promises honest:
if the behavior regresses, the docs test fails before a reader does.

(Both contracts were landed by the run-recovery workflow; they started life
as xfail placeholders in this suite and are now plain green.)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import molexp as me
from molexp.workflow import TaskContext, WorkflowCompiler, WorkflowRuntime

_boom_wf = WorkflowCompiler(name="boom")


@_boom_wf.task
async def explode(ctx: TaskContext, lr: float) -> float:
    raise RuntimeError("boom: deliberate task failure")


BOOM = _boom_wf.compile()

_ok_wf = WorkflowCompiler(name="ok")


@_ok_wf.task
async def compute(ctx: TaskContext, lr: float) -> float:
    return lr


OK = _ok_wf.compile()


def _error_txts(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "executions").rglob("error.txt"))


@pytest.mark.integration
def test_task_failure_writes_error_txt(tmp_path: Path) -> None:
    """A failed *task* persists WHY under ``executions/<exec_id>/error.txt``."""
    ws = me.Workspace(tmp_path / "lab", name="lab")
    exp = ws.project("demo").experiment("fail").run(BOOM, params={"lr": [1.0]})
    run = exp.list_runs()[0]

    async def drive() -> None:
        with run.start() as ctx:
            await WorkflowRuntime().execute(BOOM, run_context=ctx)

    asyncio.run(drive())

    assert run.status == "failed"
    error_txts = _error_txts(Path(str(run.run_dir)))
    assert error_txts, "failed run left no executions/<exec_id>/error.txt"
    body = error_txts[-1].read_text()
    assert "boom" in body, f"error.txt does not surface the failure cause:\n{body}"


@pytest.mark.integration
def test_driver_exception_writes_error_txt(tmp_path: Path) -> None:
    """An exception escaping ``with run.start()`` also lands in error.txt."""
    ws = me.Workspace(tmp_path / "lab", name="lab")
    exp = ws.project("demo").experiment("drv").run(OK, params={"lr": [1.0]})
    run = exp.list_runs()[0]

    with pytest.raises(ValueError, match="driver-side boom"), run.start():
        raise ValueError("driver-side boom")

    assert run.status == "failed"
    error_txts = _error_txts(Path(str(run.run_dir)))
    assert error_txts, "driver failure left no executions/<exec_id>/error.txt"
    assert "driver-side boom" in error_txts[-1].read_text()


@pytest.mark.integration
def test_molexp_run_exposes_documented_verbs() -> None:
    """``molexp run`` carries the flags the getting-started/guide pages teach."""
    from typer.main import get_command

    from molexp.cli import app

    run_cmd = get_command(app).commands["run"]
    opts = {opt for param in run_cmd.params for opt in getattr(param, "opts", [])}
    missing = {"--resume", "--rerun", "--fresh", "--profile", "--override"} - opts
    assert not missing, f"molexp run lost documented flags: {sorted(missing)}"
