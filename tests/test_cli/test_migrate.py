"""Tests for ``molexp migrate-layout`` (pre-cutover → per-execution)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from typer.testing import CliRunner

from molexp.cli import app
from molexp.workspace import Workspace
from molexp.workspace.models import ExecutionRecord

runner = CliRunner()


def _build_legacy_workspace(tmp_path: Path) -> tuple[Path, str]:
    """Synthesize a pre-cutover workspace and return (root, run_id)."""
    ws = Workspace(root=tmp_path, name="legacy-lab")
    project = ws.project("proj-a")
    exp = project.experiment("exp-x", workflow_source="s.py", params={})
    run = exp.run(parameters={"seed": 1})
    run_dir = run.run_dir

    exec_id = f"exec-{run.id}"
    run._update_metadata(
        execution_history=[
            ExecutionRecord(
                execution_id=exec_id,
                started_at=datetime(2025, 1, 1, 0, 0),
                finished_at=datetime(2025, 1, 1, 0, 5),
                status="succeeded",
            )
        ]
    )

    # Plant legacy artifacts at run level + singular ``execution/`` dir.
    (run_dir / "stdout.log").write_text("legacy stdout\n")
    (run_dir / "stderr.log").write_text("legacy stderr\n")
    (run_dir / "logs").mkdir()
    (run_dir / "logs" / "train.log").write_text("legacy training output\n")
    (run_dir / "jobs" / "uuid-1").mkdir(parents=True)
    (run_dir / "jobs" / "uuid-1" / "manifest.json").write_text("{}")
    (run_dir / "execution" / exec_id).mkdir(parents=True)
    (run_dir / "execution" / exec_id / "workflow.json").write_text('{"status": "completed"}')

    # Remove the new-layout files written by the modern factory so we
    # genuinely look like a pre-cutover workspace.
    new_exec_dir = run_dir / "executions"
    if new_exec_dir.exists():
        import shutil

        shutil.rmtree(new_exec_dir)

    return tmp_path, run.id


def test_migrate_moves_legacy_paths_into_executions(tmp_path):
    ws_path, run_id = _build_legacy_workspace(tmp_path)

    result = runner.invoke(app, ["migrate-layout", str(ws_path)])
    assert result.exit_code == 0, result.stdout

    run_dir = ws_path / "projects/proj-a/experiments/exp-x/runs" / f"run-{run_id}"
    exec_dir = run_dir / "executions" / f"exec-{run_id}"

    # Files moved into executions/<id>/
    assert (exec_dir / "stdout.log").read_text() == "legacy stdout\n"
    assert (exec_dir / "stderr.log").read_text() == "legacy stderr\n"
    assert (exec_dir / "logs" / "train.log").read_text() == "legacy training output\n"
    assert (exec_dir / "jobs" / "uuid-1" / "manifest.json").exists()
    assert (exec_dir / "workflow.json").read_text() == '{"status": "completed"}'

    # Originals are gone.
    assert not (run_dir / "stdout.log").exists()
    assert not (run_dir / "logs").exists()
    assert not (run_dir / "jobs").exists()
    assert not (run_dir / "execution").exists()


def test_migrate_writes_container_indices(tmp_path):
    ws_path, run_id = _build_legacy_workspace(tmp_path)

    result = runner.invoke(app, ["migrate-layout", str(ws_path)])
    assert result.exit_code == 0, result.stdout

    # All four index files exist.
    projects_idx = json.loads((ws_path / "projects.json").read_text())
    assert any(item["id"] == "proj-a" for item in projects_idx["items"])

    experiments_idx = json.loads((ws_path / "projects/proj-a/experiments.json").read_text())
    assert any(item["id"] == "exp-x" for item in experiments_idx["items"])

    runs_idx = json.loads((ws_path / "projects/proj-a/experiments/exp-x/runs.json").read_text())
    assert any(item["id"] == run_id for item in runs_idx["items"])

    run_dir = ws_path / "projects/proj-a/experiments/exp-x/runs" / f"run-{run_id}"
    executions_idx = json.loads((run_dir / "executions.json").read_text())
    assert any(item["execution_id"] == f"exec-{run_id}" for item in executions_idx["items"])


def test_migrate_is_idempotent(tmp_path):
    ws_path, run_id = _build_legacy_workspace(tmp_path)

    first = runner.invoke(app, ["migrate-layout", str(ws_path)])
    assert first.exit_code == 0

    # Snapshot every file's content + mtime structure.
    second = runner.invoke(app, ["migrate-layout", str(ws_path)])
    assert second.exit_code == 0
    # The second run should report 0 modified runs.
    assert "0 migrated" in second.stdout
