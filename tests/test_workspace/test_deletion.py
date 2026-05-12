"""Tests for Project/Experiment/Run delete APIs and catalog cascade."""

from __future__ import annotations

from datetime import datetime

import pytest

from molexp.workspace import (
    ExperimentNotFoundError,
    RunNotFoundError,
    Workspace,
)
from molexp.workspace.models import ExecutionRecord


def _build(tmp_path):
    ws = Workspace(root=tmp_path, name="lab")
    ws.materialize()
    p = ws.add_project("proj-a")
    e = p.add_experiment("exp-x", workflow_source="s.py", params={})
    r = e.add_run(parameters={"seed": 1})

    # Seed two execution dirs + history entries
    hist = []
    for i, status in enumerate(("failed", "succeeded"), start=1):
        eid = f"exec-{r.id}" if i == 1 else f"exec-{r.id}-{i}"
        (r.run_dir / "executions" / eid).mkdir(parents=True)
        hist.append(
            ExecutionRecord(
                execution_id=eid,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                status=status,
            )
        )
    r._update_metadata(execution_history=hist)
    return ws, p, e, r


class TestDeleteExecution:
    def test_removes_dir_and_history_entry(self, tmp_path):
        _ws, _p, _e, r = _build(tmp_path)
        first_exec = r.metadata.execution_history[0].execution_id
        r.delete_execution(first_exec)
        assert not (r.run_dir / "executions" / first_exec).exists()
        assert all(rec.execution_id != first_exec for rec in r.metadata.execution_history)

    def test_catalog_row_removed(self, tmp_path):
        ws, _p, _e, r = _build(tmp_path)
        first_exec = r.metadata.execution_history[0].execution_id
        # Run materialization populates catalog executions too
        r.save()
        assert any(
            row["execution_id"] == first_exec for row in ws.catalog.query_executions(run_id=r.id)
        )
        r.delete_execution(first_exec)
        assert not any(
            row["execution_id"] == first_exec for row in ws.catalog.query_executions(run_id=r.id)
        )

    def test_unknown_execution_raises(self, tmp_path):
        _ws, _p, _e, r = _build(tmp_path)
        with pytest.raises(KeyError):
            r.delete_execution("exec-does-not-exist")


class TestDeleteRun:
    def test_removes_run_dir(self, tmp_path):
        _ws, _p, e, r = _build(tmp_path)
        run_dir = r.run_dir
        assert run_dir.exists()
        e.remove_run(r.id)
        assert not run_dir.exists()

    def test_cascades_executions_in_catalog(self, tmp_path):
        ws, _p, e, r = _build(tmp_path)
        r.save()
        assert ws.catalog.query_executions(run_id=r.id)
        e.remove_run(r.id)
        assert ws.catalog.query_executions(run_id=r.id) == []
        assert ws.catalog.query_runs(experiment_id=e.id) == []

    def test_unknown_run_raises(self, tmp_path):
        _ws, _p, e, _r = _build(tmp_path)
        with pytest.raises(RunNotFoundError):
            e.remove_run("nope")


class TestDeleteExperiment:
    def test_removes_experiment_dir(self, tmp_path):
        _ws, p, e, _r = _build(tmp_path)
        exp_dir = e.experiment_dir
        assert exp_dir.exists()
        p.remove_experiment(e.id)
        assert not exp_dir.exists()

    def test_cascades_runs_and_executions(self, tmp_path):
        ws, p, e, r = _build(tmp_path)
        r.save()
        assert ws.catalog.query_runs(experiment_id=e.id)
        p.remove_experiment(e.id)
        assert ws.catalog.query_runs(experiment_id=e.id) == []
        assert ws.catalog.query_executions(run_id=r.id) == []

    def test_unknown_experiment_raises(self, tmp_path):
        _ws, p, _e, _r = _build(tmp_path)
        with pytest.raises(ExperimentNotFoundError):
            p.remove_experiment("nope")


class TestDeleteProject:
    def test_cascades_everything(self, tmp_path):
        ws, p, e, r = _build(tmp_path)
        r.save()
        ws.remove_project(p.id)
        assert not p.project_dir.exists()
        assert ws.catalog.query_runs(experiment_id=e.id) == []
        assert ws.catalog.query_executions(run_id=r.id) == []
