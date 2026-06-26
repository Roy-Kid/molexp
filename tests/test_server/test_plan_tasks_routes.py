"""Plan-task routes — PlanMode pipeline as a background server task (no LLM).

Drives ``/api/projects/{p}/experiments/{e}/plan-tasks`` with a stub gateway
factory (canned valid stage outputs), so the full PlanMode pipeline runs offline
in the app's event loop. Asserts: POST starts a task, the background run reaches
``completed``, and the generated workflow is persisted onto the experiment so the
workflow route can render it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import create_app
from molexp.server.dependencies import get_workspace
from molexp.server.plan_runtime import gateway as plan_gateway

# A WorkflowSource program that compiles to a real Workflow — ValidateWorkflowSource,
# the route's display-persist step, AND step-7 CompileWorkflow (--compile-only) all
# compile it for real.
_VALID_SOURCE = """\
from molexp.workflow import TaskContext, WorkflowCompiler


def build_workflow() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="demo")

    @wf.task
    async def build_system(ctx: TaskContext) -> dict:
        return {"structure": "system.pdb"}

    @wf.task(depends_on=["build_system"])
    async def simulate(ctx: TaskContext) -> dict:
        return {"trajectory": "traj.dcd"}

    return wf
"""

# Passes under step-7 ExecuteTests (real pytest) in the materialized layout.
_PASSING_TEST_SOURCE = """\
from generated_workflow import build_workflow


def test_build_compiles() -> None:
    assert build_workflow().compile() is not None
"""

_EXPERIMENT_REPORT = {
    "title": "Water NEMD",
    "objective": "Measure ionic mobility",
    "system_description": "SPC/E water box under an applied field",
    "experimental_design": "Apply field; record current",
}
_WORKFLOW_IR = {
    "id": "wf-water",
    "name": "water_nemd",
    "objective": "Compute mobility",
    "inputs": {},
    "tasks": [
        {
            "id": "build",
            "name": "Pack water",
            "purpose": "Build SPC/E box",
            "task_type": "molecule_builder",
            "inputs": {},
            "outputs": {"structure": "structure.pdb"},
        }
    ],
    "edges": [],
    "expected_outputs": [],
}
_BOUND_WORKFLOW = {
    "id": "bw-water",
    "workflow_ir_id": "wf-water",
    "tasks": [
        {
            "id": "b-build",
            "ir_task_id": "build",
            "capability_id": "molpy.builder.water.SPCEBuilder",
            "package": "molpy",
            "callable": "molpy.builder.water.SPCEBuilder.run",
            "parameters": {},
            "inputs": {},
            "outputs": {"structure": "structure.pdb"},
        }
    ],
    "edges": [],
    "execution_backend": "local",
    "environment": {},
    "resource_policy": {
        "backend": "local",
        "max_runtime_s": 3600,
        "denied_paths": ["/", "~/.ssh"],
    },
}
_WORKFLOW_SOURCE = {
    "source": _VALID_SOURCE,
    "module_name": "generated_workflow",
    "bound_workflow_id": "bw-water",
    "symbols": ["WorkflowCompiler", "TaskContext"],
}
_EXPERIMENT_SPEC = {
    "id": "spec-water",
    "experiment_report_id": "rep-water",
    "title": "Water NEMD",
    "objective": "Measure ionic mobility",
    "variables": [],
    "controlled_conditions": [],
    "resolved_questions": [],
    "assumptions": [],
}
_INPUT_SET = {
    "id": "is-water",
    "experiment_spec_id": "spec-water",
    "title": "single-cell sweep",
    "sweep_axes": [],
    "strategy": "grid",
    "total_runs": 1,
}
_TEST_SPEC = {
    "id": "tsb-water",
    "bound_workflow_id": "bw-water",
    "specs": [
        {
            "id": "ts-water",
            "name": "workflow_compiles",
            "kind": "unit_test",
            "target_task_id": "build",
            "description": "The generated workflow module compiles into a workflow.",
        }
    ],
}
_TEST_SOURCE = {
    "source": _PASSING_TEST_SOURCE,
    "module_name": "test_generated_workflow",
    "test_spec_id": "ts-water",
    "bound_workflow_id": "bw-water",
    "symbols": ["build_workflow"],
}


def _stub_factory(run: Any, model: str) -> Any:
    from molexp.harness.gateways.stub import StubAgentGateway
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    store = FileArtifactStore(root=Path(run.run_dir / "artifacts"))
    gw = StubAgentGateway(artifact_store=store)
    gw.register(
        agent_name="experiment_report_writer",
        output=_EXPERIMENT_REPORT,
        output_kind="experiment_report",
    )
    gw.register(
        agent_name="experiment_spec_generator",
        output=_EXPERIMENT_SPEC,
        output_kind="experiment_spec",
    )
    gw.register(agent_name="workflow_ir_extractor", output=_WORKFLOW_IR, output_kind="workflow_ir")
    gw.register(
        agent_name="bound_workflow_binder", output=_BOUND_WORKFLOW, output_kind="bound_workflow"
    )
    gw.register(
        agent_name="workflow_source_writer",
        output=_WORKFLOW_SOURCE,
        output_kind="workflow_source",
    )
    gw.register(
        agent_name="plan_reviewer",
        output={"passed": True, "findings": [], "summary": "faithful"},
        output_kind="plan_review",
    )
    gw.register(agent_name="test_spec_writer", output=_TEST_SPEC, output_kind="test_spec")
    gw.register(agent_name="test_code_writer", output=_TEST_SOURCE, output_kind="test_source")
    gw.register(agent_name="input_set_generator", output=_INPUT_SET, output_kind="input_set")
    return gw


@pytest.fixture
def plan_client(workspace: Any, project: Any) -> Iterator[TestClient]:
    project.add_experiment("plan-exp")
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: workspace
    plan_gateway.set_plan_gateway_factory(_stub_factory)
    with TestClient(app) as client:  # context manager → lifespan cancels tasks on teardown
        yield client
    plan_gateway.reset_plan_gateway_factory()


_BASE = "/api/projects/test-project/experiments/plan-exp/plan-tasks"


def _await_terminal(client: TestClient, task_id: str, tries: int = 400) -> dict[str, Any]:
    """Poll the task until it leaves ``running`` (gives the bg loop time to run).

    The 9-step pipeline now spawns real pytest + compile subprocesses at step 7,
    so allow a generous budget (≈20s) before giving up.
    """
    url = f"{_BASE}/{task_id}"
    body: dict[str, Any] = client.get(url).json()
    for _ in range(tries):
        if body["status"] != "running":
            return body
        time.sleep(0.05)
        body = client.get(url).json()
    return body


def test_create_plan_task_runs_pipeline_and_persists_workflow(plan_client: TestClient) -> None:
    # ground=false keeps the test offline (no molmcp spawn).
    resp = plan_client.post(
        _BASE, json={"draft": "screen solvent ratios", "model": "stub-model", "ground": False}
    )
    assert resp.status_code == 201, resp.text
    started = resp.json()
    assert started["status"] == "running"
    assert started["runId"]

    final = _await_terminal(plan_client, started["taskId"])
    assert final["status"] == "completed", final
    assert final["workflowPersisted"] is True

    # Same run is listed.
    listed = plan_client.get(_BASE).json()
    assert any(t["taskId"] == started["taskId"] for t in listed["tasks"])

    # The plan-generated workflow now renders through the existing workflow route.
    wf = plan_client.get("/api/projects/test-project/experiments/plan-exp/workflow")
    assert wf.status_code == 200, wf.text
    task_ids = {t["task_id"] for t in wf.json()["document"]["task_configs"]}
    assert {"build_system", "simulate"} <= task_ids


def test_create_plan_task_is_idempotent_on_same_draft(plan_client: TestClient) -> None:
    payload = {"draft": "same draft text", "model": "stub-model", "ground": False}
    first = plan_client.post(_BASE, json=payload).json()
    _await_terminal(plan_client, first["taskId"])
    second = plan_client.post(_BASE, json=payload).json()
    # Content-addressed run id: same draft → same run (the ledger resumes).
    assert second["runId"] == first["runId"]


def test_get_unknown_plan_task_returns_404(plan_client: TestClient) -> None:
    assert plan_client.get(f"{_BASE}/plan-does-not-exist").status_code == 404


def test_grounded_plan_task_threads_capability_registry(
    plan_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ground=true resolves a registry and threads it into PlanMode/validation.

    The stub binds ``molpy.builder.water.SPCEBuilder``; a registry that knows it
    makes ValidateBoundWorkflow's capability checks fire and pass — proving the
    grounding path runs end to end without spawning molmcp.
    """
    called: list[str] = []

    async def _fake_aresolve(workspace_root: str, **_: Any) -> Any:
        from molexp.harness import InMemoryCapabilityRegistry
        from molexp.harness.schemas import ToolCapability

        called.append(workspace_root)
        return InMemoryCapabilityRegistry(
            [
                ToolCapability(
                    id="molpy.builder.water.SPCEBuilder",
                    package="molpy",
                    name="SPCEBuilder",
                    description="Build an SPC/E water box.",
                    input_schema={"type": "object"},
                    output_schema={},
                    callable_path="molpy.builder.water.SPCEBuilder",
                    supported_backends=["local"],
                    tags=["class"],
                )
            ]
        )

    monkeypatch.setattr("molexp.mcp_capabilities.aresolve_capability_registry", _fake_aresolve)

    resp = plan_client.post(_BASE, json={"draft": "grounded run", "model": "stub-model"})
    assert resp.status_code == 201, resp.text
    final = _await_terminal(plan_client, resp.json()["taskId"])
    assert final["status"] == "completed", final
    assert final["workflowPersisted"] is True
    assert called, "grounding resolver was not invoked for ground=true"
