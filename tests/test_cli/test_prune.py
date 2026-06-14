"""Tests for ``molexp runs prune`` hierarchical cleanup."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from molexp.cli import app
from molexp.workspace import Workspace
from molexp.workspace.models import ExecutionRecord

runner = CliRunner()


@pytest.fixture
def seeded_workspace(tmp_path):
    ws = Workspace(root=tmp_path, name="prune-lab")
    project = ws.add_project("proj-a")
    exp = project.add_experiment("exp-x", workflow_source="s.py", params={})
    run = exp.add_run(params={"seed": 1})

    # Seed three executions: two failed, one succeeded.
    history = []
    for i, status in enumerate(("succeeded", "failed", "failed"), start=1):
        exec_id = f"exec-{run.id}" if i == 1 else f"exec-{run.id}-{i}"
        exec_dir = Path(run.run_dir / "executions" / exec_id)
        exec_dir.mkdir(parents=True, exist_ok=True)
        (exec_dir / "workflow.json").write_text('{"status":"' + status + '"}')
        history.append(
            ExecutionRecord(
                execution_id=exec_id,
                started_at=datetime(2025, 4, 1, 10, i),
                finished_at=datetime(2025, 4, 1, 10, i + 1),
                status=status,
            )
        )
    run._update_metadata(execution_history=history, status="succeeded")
    return tmp_path, run


def test_prune_deletes_failed_executions(seeded_workspace):
    ws_path, run = seeded_workspace

    # Pick project 1 → experiment 1 → run 1 → executions "failed"
    # (multi-select by status keyword) → confirm "y".
    result = runner.invoke(
        app,
        ["runs", "prune", "--path", str(ws_path)],
        input="1\n1\n1\nfailed\ny\n",
    )

    assert result.exit_code == 0, result.stdout

    # Two failed dirs gone, the succeeded one remains.
    exec_root = Path(run.run_dir / "executions")
    remaining = sorted(p.name for p in exec_root.iterdir())
    assert remaining == [f"exec-{run.id}"]

    # Reload metadata and check history.
    from molexp.workspace import Workspace as _WS

    reloaded = _WS.load(ws_path).get_project("proj-a").get_experiment("exp-x").get_run(run.id)
    assert [e.execution_id for e in reloaded.metadata.execution_history] == [f"exec-{run.id}"]


def test_prune_abort_on_empty_selection(seeded_workspace):
    ws_path, run = seeded_workspace

    result = runner.invoke(
        app,
        ["runs", "prune", "--path", str(ws_path)],
        input="1\n1\n1\n\n",  # empty selection at layer 4
    )

    assert result.exit_code == 0
    # All 3 dirs intact.
    exec_root = Path(run.run_dir / "executions")
    assert len(list(exec_root.iterdir())) == 3


def test_prune_range_syntax(seeded_workspace):
    ws_path, run = seeded_workspace

    # select 2-3 (the two failed entries), confirm.
    result = runner.invoke(
        app,
        ["runs", "prune", "--path", str(ws_path)],
        input="1\n1\n1\n2-3\ny\n",
    )

    assert result.exit_code == 0, result.stdout
    exec_root = Path(run.run_dir / "executions")
    remaining = sorted(p.name for p in exec_root.iterdir())
    assert remaining == [f"exec-{run.id}"]


def test_prune_refuses_live_running_record(tmp_path):
    ws = Workspace(root=tmp_path, name="live-lab")
    project = ws.add_project("proj-a")
    exp = project.add_experiment("exp-x", workflow_source="s.py", params={})
    run = exp.add_run(params={})

    exec_id = f"exec-{run.id}"
    Path(run.run_dir / "executions" / exec_id).mkdir(parents=True)
    run._update_metadata(
        status="running",
        execution_history=[
            ExecutionRecord(
                execution_id=exec_id,
                started_at=datetime.now(),
                status="running",
            )
        ],
    )

    result = runner.invoke(
        app,
        ["runs", "prune", "--path", str(tmp_path)],
        input="1\n1\n1\n1\n",
    )

    assert result.exit_code == 1
    assert "Refusing" in result.stdout
    assert Path(run.run_dir / "executions" / exec_id).exists()
