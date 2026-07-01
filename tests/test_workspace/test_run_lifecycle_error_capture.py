"""RED tests: a FAILED run must persist WHY it failed in the canonical record.

When a task raises, the workflow engine catches the exception into a failed
``WorkflowResult`` and does NOT re-raise (so ``molexp run`` can resume). On that
path ``RunContext.__exit__`` set ``status=failed`` but dropped the error message
that ``mark_failed`` had already stashed on the context — leaving
``run.json``/``execution.json`` with ``error: null`` while the reason lived only
in the workflow-layer ``workflow.json``. That is a silent-invalid-state defect
(the authoritative record lies) and the reason a beginner's failed run surfaced
no "why" through any CLI command.

These tests pin: the failing-task error (type + message) lands in the
workspace-owned canonical record, and the P0.2 context surfaces it on the
``failed_run`` health flag.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import molexp as me
from molexp.workflow import WorkflowRuntime, promote_callable
from molexp.workspace.workspace_context import assemble_workspace_context


def _boom(inputs, config):
    return 1 / 0  # ZeroDivisionError: float division by zero


def _failing_spec():
    return promote_callable(_boom, name="boom")


def _run_failing(tmp_path: Path):
    ws = me.Workspace(tmp_path / "ws", name="ws")
    exp = ws.add_project("demo").add_experiment("train")
    run = exp.add_run(params={"seed": 0})
    with run.start() as ctx:
        asyncio.run(WorkflowRuntime().execute(_failing_spec(), run_context=ctx))
    # Reload fresh so we read what was persisted, not the in-memory instance.
    reloaded = ws.get_project("demo").get_experiment("train").get_run(run.id)
    return ws, reloaded


def test_failed_run_captures_error_into_run_metadata(tmp_path: Path) -> None:
    _ws, run = _run_failing(tmp_path)

    assert run.status == "failed"
    err = run.metadata.error
    assert err is not None, "a FAILED run must persist its error in run.json (was null)"
    assert "division by zero" in err.message
    assert err.type == "ZeroDivisionError", "the exception type must survive, not a placeholder"


def test_failed_run_captures_error_into_execution_metadata(tmp_path: Path) -> None:
    import json

    _ws, run = _run_failing(tmp_path)

    history = run.execution_history
    assert history, "the run must have an execution record"
    exec_id = history[-1].execution_id
    data = json.loads((Path(run.run_dir) / "executions" / exec_id / "execution.json").read_text())
    assert data.get("error") is not None, "execution.json error must not be null on a failed run"
    assert "division by zero" in data["error"]["message"]


def test_failed_run_reason_surfaced_on_context_health_flag(tmp_path: Path) -> None:
    ws, run = _run_failing(tmp_path)

    ctx = assemble_workspace_context(ws)
    flags = [f for f in ctx.stale_or_missing if f.kind == "failed_run" and run.id in f.ref]
    assert flags, "a failed run must raise a failed_run health flag"
    assert "division by zero" in flags[0].detail, (
        "the health flag must say WHY the run failed, not just that it did"
    )


def test_runs_info_prints_the_failure_reason_and_retry_hint(tmp_path: Path) -> None:
    import re

    from typer.testing import CliRunner

    from molexp.cli import app

    ws, run = _run_failing(tmp_path)
    result = CliRunner().invoke(app, ["runs", "info", "demo", "train", run.id, "-t", str(ws.root)])
    assert result.exit_code == 0, result.output
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "ZeroDivisionError" in plain, plain
    assert "division by zero" in plain, plain
    assert "--resume" in plain, plain
