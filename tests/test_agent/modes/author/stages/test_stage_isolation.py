"""Per-stage isolation tests for AuthorMode (ac-007)."""

from __future__ import annotations

from pathlib import Path

from molexp.agent.modes.author._mode import AuthorMode
from molexp.agent.modes.author.stages import (
    CompileTaskIR,
    GenerateTaskImplementations,
    GenerateTaskTests,
    GenerateWorkflowSkeleton,
    LowerPlanGraph,
    RunTaskDebugLoop,
    ValidateWorkspace,
    WriteManifest,
)
from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.workspace import Workspace


def _minimal_author_mode(tmp_path: Path) -> AuthorMode:
    ws = Workspace(tmp_path / "lab")
    pf = ws.add_folder(PlanFolder(name="plan-1"))
    return AuthorMode(plan_folder=pf)


def test_author_mode_pipeline_carries_eight_stages_in_order(tmp_path: Path) -> None:
    mode = _minimal_author_mode(tmp_path)
    stages = mode.pipeline.stages
    assert len(stages) == 8
    expected_classes = (
        LowerPlanGraph,
        CompileTaskIR,
        GenerateWorkflowSkeleton,
        GenerateTaskTests,
        GenerateTaskImplementations,
        RunTaskDebugLoop,
        ValidateWorkspace,
        WriteManifest,
    )
    for stage, cls in zip(stages, expected_classes, strict=True):
        assert isinstance(stage, cls)
    assert tuple(s.name for s in stages) == (
        "LowerPlanGraph",
        "CompileTaskIR",
        "GenerateWorkflowSkeleton",
        "GenerateTaskTests",
        "GenerateTaskImplementations",
        "RunTaskDebugLoop",
        "ValidateWorkspace",
        "WriteManifest",
    )
