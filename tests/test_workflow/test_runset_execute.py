"""``RunSet.execute`` — batch tracked execution over a sweep (runset-api sub-task 2).

Locks the design red line: every run goes through the SAME execution path as
``molexp run`` (RunContext lifecycle → status machine, ``_ops`` sidecar,
heartbeat → WorkflowRuntime), one run's failure never sinks its siblings, and
the summary reports statuses honestly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import molexp as me
from molexp.workflow import WorkflowCompiler
from molexp.workspace.models import RunStatus
from molexp.workspace.runset import RunSet


@pytest.fixture
def experiment(tmp_path: Path):
    ws = me.Workspace(tmp_path / "ws", name="lab")
    return ws.add_project("demo").add_experiment("scan")


def _build_wf(calls: list | None = None) -> WorkflowCompiler:
    wf = WorkflowCompiler(name="scan")

    @wf.task
    def simulate(lr: float, batch: int) -> float:
        if calls is not None:
            calls.append((lr, batch))
        return lr * batch

    @wf.task(depends_on=["simulate"])
    def score(simulate: float) -> float:
        return 1.0 / simulate

    return wf


class TestRunSetExecute:
    def test_full_sweep_executes_and_summarizes(self, experiment) -> None:
        rs = experiment.sweep(_build_wf(), {"lr": [0.1, 0.2], "batch": [10, 20]})
        summary = rs.execute()
        records = summary.to_records()
        assert len(records) == 4
        assert all(rec["status"] == RunStatus.SUCCEEDED.value for rec in records)
        # params flattened + per-task outputs + identity columns.
        by_cell = {(rec["lr"], rec["batch"]): rec for rec in records}
        assert by_cell[(0.2, 10)]["simulate"] == pytest.approx(2.0)
        assert by_cell[(0.2, 10)]["score"] == pytest.approx(0.5)
        assert all(rec["run_id"] for rec in records)
        # min/max over a task output column work end-to-end.
        assert summary.max_by("simulate")["lr"] == 0.2
        assert summary.max_by("simulate")["batch"] == 20

    def test_failed_cell_does_not_sink_siblings(self, experiment) -> None:
        # lr == 0.0 → simulate returns 0.0 → score divides by zero.
        rs = experiment.sweep(_build_wf(), {"lr": [0.0, 0.2], "batch": [10]})
        summary = rs.execute()
        records = {rec["lr"]: rec for rec in summary.to_records()}
        assert records[0.2]["status"] == RunStatus.SUCCEEDED.value
        assert records[0.0]["status"] == RunStatus.FAILED.value
        assert records[0.0]["error"] and "ZeroDivisionError" in records[0.0]["error"]
        assert [rec.run_id for rec in summary.failed] == [records[0.0]["run_id"]]

    def test_execute_skips_non_pending_runs(self, experiment) -> None:
        calls: list = []
        rs = experiment.sweep(_build_wf(calls), {"lr": [0.1, 0.2], "batch": [10]})
        rs.execute()
        assert len(calls) == 2
        # Second execute: everything already succeeded → nothing recomputed,
        # summary still reports all four columns from persisted node outputs.
        summary = rs.execute()
        assert len(calls) == 2
        records = summary.to_records()
        assert all(rec["status"] == RunStatus.SUCCEEDED.value for rec in records)
        assert sorted(rec["simulate"] for rec in records) == [
            pytest.approx(1.0),
            pytest.approx(2.0),
        ]

    def test_parallel_execution(self, experiment) -> None:
        rs = experiment.sweep(_build_wf(), {"lr": [0.1, 0.2], "batch": [10, 20]})
        summary = rs.execute(parallel=2)
        assert all(rec["status"] == RunStatus.SUCCEEDED.value for rec in summary.to_records())

    def test_parallel_must_be_positive(self, experiment) -> None:
        rs = experiment.sweep(_build_wf(), {"lr": [0.1], "batch": [10]})
        with pytest.raises(ValueError, match="parallel"):
            rs.execute(parallel=0)

    def test_collect_reads_back_in_fresh_runset(self, experiment) -> None:
        """A later session can rebuild the RunSet from disk and read the
        persisted per-task outputs without re-executing anything."""
        experiment.sweep(_build_wf(), {"lr": [0.1, 0.2], "batch": [10]}).execute()
        reloaded = experiment.runs()
        assert isinstance(reloaded, RunSet)
        records = reloaded.collect().to_records()
        assert len(records) == 2
        assert all(rec["status"] == RunStatus.SUCCEEDED.value for rec in records)
        assert sorted(rec["simulate"] for rec in records) == [
            pytest.approx(1.0),
            pytest.approx(2.0),
        ]
