"""Per-stage isolation tests for RunMode (ac-007)."""

from __future__ import annotations

from pathlib import Path

from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.agent.modes.run._mode import RunMode
from molexp.agent.modes.run.stages import (
    ExecuteWorkflow,
    LoadMaterializedWorkflow,
    RepairRuntimeFailure,
)
from molexp.workspace import Workspace


def _minimal_run_mode(tmp_path: Path) -> RunMode:
    ws = Workspace(tmp_path / "lab")
    experiment = ws.add_project("proj").add_experiment("exp")
    pf = ws.add_folder(PlanFolder(name="plan-1"))
    return RunMode(plan_folder=pf, experiment=experiment)


def test_run_mode_pipeline_carries_three_stages_in_order(tmp_path: Path) -> None:
    mode = _minimal_run_mode(tmp_path)
    stages = mode.pipeline.stages
    assert len(stages) == 3
    assert isinstance(stages[0], LoadMaterializedWorkflow)
    assert isinstance(stages[1], ExecuteWorkflow)
    assert isinstance(stages[2], RepairRuntimeFailure)
    assert tuple(s.name for s in stages) == (
        "LoadMaterializedWorkflow",
        "ExecuteWorkflow",
        "RepairRuntimeFailure",
    )
