"""``Experiment.sweep`` / ``Experiment.runs`` — RunSet-returning sugar (runset-api).

``sweep`` reuses the exact seeding path of ``Experiment.run`` (idempotent
content-addressed ``add_runs``) and the same cross-layer ``WorkflowExecutor``
seam for the workflow association; the only new behaviour is the returned
:class:`~molexp.workspace.runset.RunSet` and the dict→GridSpace upgrade.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import molexp as me
from molexp.workflow import WorkflowCompiler, default_binding_registry
from molexp.workspace import GridSpace
from molexp.workspace.runset import RunSet


@pytest.fixture
def experiment(tmp_path: Path):
    ws = me.Workspace(tmp_path / "ws", name="lab")
    return ws.add_project("demo").add_experiment("scan")


def _build_wf() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="scan")

    @wf.task
    def cell(lr: float, batch: int) -> float:
        return lr * batch

    return wf


class TestSweep:
    def test_dict_upgrades_to_grid_and_returns_runset(self, experiment) -> None:
        rs = experiment.sweep(_build_wf().compile(), {"lr": [0.1, 0.2], "batch": [16, 32]})
        assert isinstance(rs, RunSet)
        assert len(rs) == 4
        cells = {(r.parameters["lr"], r.parameters["batch"]) for r in rs}
        assert cells == {(0.1, 16), (0.1, 32), (0.2, 16), (0.2, 32)}

    def test_sweep_is_idempotent(self, experiment) -> None:
        compiled = _build_wf().compile()
        first = experiment.sweep(compiled, {"lr": [0.1, 0.2]})
        second = experiment.sweep(compiled, {"lr": [0.1, 0.2]})
        assert sorted(r.id for r in first) == sorted(r.id for r in second)
        assert len(experiment.list_runs()) == 2

    def test_sweep_scalar_axis_fails_fast(self, experiment) -> None:
        with pytest.raises(ValueError, match="batch"):
            experiment.sweep(_build_wf().compile(), {"lr": [0.1], "batch": 16})

    def test_sweep_auto_compiles_compiler(self, experiment) -> None:
        rs = experiment.sweep(_build_wf(), {"lr": [0.1]})
        assert len(rs) == 1
        bound = default_binding_registry.for_experiment(experiment)
        assert bound is not None, "sweep must bind the (auto-)compiled workflow"

    def test_sweep_binds_workflow_for_cli_discovery(self, experiment) -> None:
        compiled = _build_wf().compile()
        experiment.sweep(compiled, {"lr": [0.1]})
        assert default_binding_registry.for_experiment(experiment) is compiled


class TestRuns:
    def test_runs_wraps_existing_runs(self, experiment) -> None:
        experiment.add_runs(GridSpace({"lr": [0.1, 0.2]}))
        rs = experiment.runs()
        assert isinstance(rs, RunSet)
        assert len(rs) == 2
        assert {r.id for r in rs} == {r.id for r in experiment.list_runs()}

    def test_runs_empty_experiment(self, experiment) -> None:
        assert len(experiment.runs()) == 0
