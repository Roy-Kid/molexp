"""End-to-end test for the 10-node PlanMode pipeline.

Covers acceptance criterion ac-006: the pipeline runs every node,
the validation report writes, the manifest gets ``status: approved``
plus a ``handoff:`` block, and ``mode_state["plan"]["handoff"]``
decodes to a valid :class:`PlanRunHandoff`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.mode import AgentRunResult
from molexp.agent.modes.plan import (
    PlanMode,
    PlanRunHandoff,
    PlanWorkspaceHandle,
)
from molexp.agent.session import AgentSession
from molexp.workspace import Workspace

from .conftest import FakeProvider


@pytest.fixture
def full_pipeline_handle(tmp_path: Path) -> PlanWorkspaceHandle:
    return PlanWorkspaceHandle.materialize(Workspace(tmp_path / "ws"), plan_id="full_pipeline")


@pytest.mark.asyncio
async def test_pipeline_full_run_through_human_review(
    full_pipeline_handle: PlanWorkspaceHandle,
) -> None:
    provider = FakeProvider()
    mode = PlanMode(workspace_handle=full_pipeline_handle, provider=provider)  # type: ignore[arg-type]
    session = AgentSession()
    result = await mode.run(
        harness=None,  # type: ignore[arg-type]
        session=session,
        user_input="Investigate Suzuki coupling.",
    )

    assert isinstance(result, AgentRunResult)
    assert result.mode_state is not None
    plan_compat = result.mode_state["plan"]
    assert plan_compat["approved"] is True
    handoff_dict = plan_compat["handoff"]
    assert isinstance(handoff_dict, dict)
    rebuilt = PlanRunHandoff.model_validate(handoff_dict)
    assert rebuilt.plan_id == "full_pipeline"

    # Validation report wrote.
    assert full_pipeline_handle.validation_report_path().exists()

    # Every per-task module + test exists.
    for task_id in ("prepare", "couple", "isolate"):
        assert (full_pipeline_handle.tasks_pkg_dir() / f"{task_id}.py").exists()
        assert (full_pipeline_handle.tests_dir() / f"test_{task_id}.py").exists()
    # Topology-pin test exists.
    assert (full_pipeline_handle.tests_dir() / "test_workflow_structure.py").exists()


@pytest.mark.asyncio
async def test_pipeline_full_records_per_node_outputs(
    full_pipeline_handle: PlanWorkspaceHandle,
) -> None:
    """Every one of the 10 pipeline nodes lands its *Result in outputs."""
    provider = FakeProvider()
    mode = PlanMode(workspace_handle=full_pipeline_handle, provider=provider)  # type: ignore[arg-type]
    result = await mode.run(
        harness=None,  # type: ignore[arg-type]
        session=AgentSession(),
        user_input="report",
    )
    outputs = result.mode_state["outputs"]
    expected_nodes = {
        "IngestReport",
        "DraftReportDigest",
        "DraftImplementationPlan",
        "CompileWorkflowIR",
        "CompileTaskIR",
        "GenerateWorkflowSkeleton",
        "GenerateTaskTests",
        "GenerateTaskImplementations",
        "ValidateWorkspace",
        "HumanReview",
    }
    assert expected_nodes.issubset(outputs.keys())
