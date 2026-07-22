"""RunSet / RunSetResult — the workspace-layer batch container (runset-api sub-task 2).

Pure-container behaviour only: sequence protocol, record building, summary
helpers (``to_records`` / ``min_by`` / ``max_by``), and the unwired-executor
failure mode. Execution through the workflow layer is locked in
``tests/test_workflow/test_runset_execute.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from molexp.workspace import GridSpace, Workspace
from molexp.workspace.runset import RunRecord, RunSet, RunSetResult


@pytest.fixture
def experiment(tmp_path: Path):
    ws = Workspace(root=tmp_path / "ws", name="lab")
    return ws.add_project("demo").add_experiment("sweep")


def _seeded_runset(experiment) -> RunSet:
    runs = experiment.add_runs(GridSpace({"lr": [0.1, 0.2], "batch": [16, 32]}))
    return RunSet(runs)


class TestRunSetContainer:
    def test_sequence_protocol(self, experiment) -> None:
        rs = _seeded_runset(experiment)
        assert len(rs) == 4
        assert rs[0].parameters == {"lr": 0.1, "batch": 16}
        assert [r.id for r in rs] == [r.id for r in rs.runs]

    def test_collect_unexecuted_runs_reports_pending(self, experiment) -> None:
        summary = _seeded_runset(experiment).collect()
        assert len(summary) == 4
        assert all(rec["status"] == "pending" for rec in summary.to_records())

    def test_execute_without_workflow_layer_fails_fast(self, tmp_path: Path) -> None:
        """Without ``import molexp.workflow`` the executor seam is unwired;
        ``RunSet.execute`` must fail fast with guidance, never fall back."""
        code = (
            "import sys\n"
            "from molexp.workspace import GridSpace, Workspace\n"
            "from molexp.workspace.runset import RunSet\n"
            "assert 'molexp.workflow' not in sys.modules\n"
            f"ws = Workspace(root={str(tmp_path / 'ws2')!r}, name='lab')\n"
            "exp = ws.add_project('p').add_experiment('e')\n"
            "rs = RunSet(exp.add_runs(GridSpace({'x': [1]})))\n"
            "try:\n"
            "    rs.execute()\n"
            "except RuntimeError as exc:\n"
            "    assert 'workflow' in str(exc), str(exc)\n"
            "else:\n"
            "    raise SystemExit('RunSet.execute did not fail fast')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr or result.stdout


def _result_fixture() -> RunSetResult:
    return RunSetResult(
        entries=(
            RunRecord(
                run_id="r1",
                status="succeeded",
                params={"lr": 0.1},
                outputs={"loss": 0.5},
            ),
            RunRecord(
                run_id="r2",
                status="succeeded",
                params={"lr": 0.2},
                outputs={"loss": 0.3},
            ),
            RunRecord(
                run_id="r3",
                status="failed",
                params={"lr": 0.4},
                outputs={},
                error="ZeroDivisionError: bad cell",
            ),
        )
    )


class TestRunSetResult:
    def test_to_records_flattens_params_and_outputs(self) -> None:
        records = _result_fixture().to_records()
        assert records[0] == {
            "run_id": "r1",
            "status": "succeeded",
            "error": None,
            "lr": 0.1,
            "loss": 0.5,
        }
        assert records[2]["status"] == "failed"
        assert "ZeroDivisionError" in records[2]["error"]

    def test_reserved_keys_win_over_collisions(self) -> None:
        result = RunSetResult(
            entries=(
                RunRecord(
                    run_id="r1",
                    status="succeeded",
                    params={"status": "sneaky", "run_id": "fake"},
                    outputs={},
                ),
            )
        )
        record = result.to_records()[0]
        assert record["run_id"] == "r1"
        assert record["status"] == "succeeded"

    def test_min_by_and_max_by(self) -> None:
        result = _result_fixture()
        assert result.min_by("loss")["run_id"] == "r2"
        assert result.max_by("loss")["run_id"] == "r1"

    def test_min_by_missing_key_raises(self) -> None:
        with pytest.raises(ValueError, match="nope"):
            _result_fixture().min_by("nope")

    def test_no_dataframe_bridge(self) -> None:
        """molexp deliberately ships NO pandas bridge — ``to_records()`` rows
        are plain dicts for whatever analysis stack the operator uses."""
        assert not hasattr(_result_fixture(), "to_dataframe")
