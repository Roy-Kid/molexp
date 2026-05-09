"""End-to-end test for the 11-node PlanMode pipeline.

Covers acceptance criterion ac-006: the pipeline runs every node,
the validation report writes, the manifest gets ``status: ready_for_run``
plus a ``handoff:`` block, and ``mode_state["plan"]["handoff"]``
decodes to a valid :class:`PlanRunHandoff`.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from molexp.agent.mode import AgentRunResult
from molexp.agent.modes.plan import (
    PlanMode,
    PlanRunHandoff,
    PlanWorkspaceHandle,
)
from molexp.agent.modes.plan.schemas import TaskImplementationModule
from molexp.agent.session import AgentSession
from molexp.workflow import Workflow
from molexp.workspace import Workspace

from .conftest import FakeProvider, canned_presets


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
    assert plan_compat["ready_for_run"] is True
    assert plan_compat["status"] == "ready_for_run"
    handoff_dict = plan_compat["handoff"]
    assert isinstance(handoff_dict, dict)
    rebuilt = PlanRunHandoff.model_validate(handoff_dict)
    assert rebuilt.plan_id == "full_pipeline"
    assert rebuilt.entrypoint_symbol == "create_workflow"

    module = _import_generated_workflow(full_pipeline_handle)
    workflow = module.create_workflow()
    assert isinstance(workflow, Workflow)

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
    """Every one of the 11 pipeline nodes lands its *Result in outputs."""
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
        "FinalHandoffCheck",
    }
    assert expected_nodes.issubset(outputs.keys())


@pytest.mark.asyncio
async def test_auto_approved_but_non_importable_workspace_is_not_ready_for_run(
    full_pipeline_handle: PlanWorkspaceHandle,
) -> None:
    presets: dict[type[BaseModel], object] = canned_presets()
    impls = dict(presets[TaskImplementationModule])  # type: ignore[arg-type]
    impls["prepare"] = TaskImplementationModule(
        task_id="prepare",
        source=(
            '"""Missing the expected Prepare class."""\n\n'
            "from molexp.workflow import Task\n\n\n"
            "class NotPrepare(Task):\n"
            "    async def execute(self, ctx):\n"
            "        return None\n"
        ),
    )
    presets[TaskImplementationModule] = impls
    mode = PlanMode(
        workspace_handle=full_pipeline_handle,
        provider=FakeProvider(presets=presets),  # type: ignore[arg-type]
    )
    result = await mode.run(
        harness=None,  # type: ignore[arg-type]
        session=AgentSession(),
        user_input="report",
    )

    plan = result.mode_state["plan"]
    assert plan["approved"] is True
    assert plan["ready_for_run"] is False
    assert plan["status"] == "validation_failed"
    assert "handoff_entrypoint_imports" in full_pipeline_handle.validation_report_path().read_text()


def _import_generated_workflow(handle: PlanWorkspaceHandle):
    source_root = str(handle.src_dir())
    root_name = "experiment"
    old_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == root_name or name.startswith(root_name + ".")
    }
    for name in old_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, source_root)
    try:
        return importlib.import_module("experiment.workflow")
    finally:
        sys.path.remove(source_root)
        for name in list(sys.modules):
            if name == root_name or name.startswith(root_name + "."):
                sys.modules.pop(name, None)
        sys.modules.update(old_modules)
