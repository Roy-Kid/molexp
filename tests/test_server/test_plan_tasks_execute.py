"""RED tests — plan execute tail over HTTP (spec vision-loop-02-plan-execute-server).

``POST /projects/{p}/experiments/{e}/plan-tasks`` gains ``execute: bool = False``
and ``compute_target: str | None`` (a *named* workspace target); the response
gains ``execute: bool``. An ``execute=true`` task threads
``PlanMode(execute=True, compute_target=<resolved ComputeTarget>)`` and rides
the slice-01 approvals inbox through THREE gates
(``approve-experiment-spec`` → ``approve-plan`` → ``approve-execution``);
after the grants, ``GET .../plans/{run_id}`` carries non-null ``finalReport``
+ ``auditReport``. An unknown ``compute_target`` name → 422 listing the known
target names (fail-fast, no fallback).

Offline + deterministic: the stub gateway factory serves canned stage outputs
(same idiom as ``test_plan_tasks_routes.py``) and the executor seam is patched
to ``DryRunExecutor`` — PlanMode's default executor is constructed from the
module-global ``LocalExecutor`` in ``harness/modes/plan.py`` (the server task
keeps the default per the spec), so patching that symbol keeps step 7 AND the
execute tail free of real subprocesses.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import create_app
from molexp.server.dependencies import get_workspace
from molexp.services.plan_runtime import gateway as plan_gateway
from molexp.workspace.models import ComputeTarget
from molexp.workspace.targets import add_target

if TYPE_CHECKING:
    from collections.abc import Iterator

    from molexp.harness.gateways.stub import StubAgentGateway
    from molexp.workspace import Workspace
    from molexp.workspace.project import Project
    from molexp.workspace.run import Run

# ─────────────────────────────────────────────── canned stage outputs
# Same canned pipeline payloads as test_plan_tasks_routes.py (kept local —
# test files are append-only under RED), plus the execute-tail final report.

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
# The execute tail's GenerateFinalReport is an LLM stage (`final_report_writer`)
# — canned like the CLI execute-tail test; GenerateAuditReport is deterministic.
_FINAL_REPORT = {
    "title": "CannedServerExecuteFinalReport",
    "objective": "Measure ionic mobility from real execution outputs.",
    "methods_summary": "Canned workflow driven through the harness executor seam.",
    "test_summary": "Generated unit test compiled the workflow (dry-run executor).",
    "execution_summary": "Driver dry-run reported exit 0.",
    "results": "No numerical outputs under DryRunExecutor.",
    "conclusions": "Server-side plan-then-execute wiring verified end-to-end.",
    "limitations": ["dry-run executor"],
    "next_steps": ["re-run with the default LocalExecutor"],
}

# The three plan gates, in pipeline order — request ids are deterministic
# (PlanMode._spec_request / _final_request / _execution_request).
_SPEC_GATE = "approve-experiment-spec"
_PLAN_GATE = "approve-plan"
_EXECUTION_GATE = "approve-execution"


def _stub_factory(run: Run, model: str) -> StubAgentGateway:
    from pathlib import Path

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
    gw.register(agent_name="final_report_writer", output=_FINAL_REPORT, output_kind="final_report")
    return gw


@pytest.fixture
def plan_client(
    workspace: Workspace, project: Project, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    from molexp.harness import DryRunExecutor

    project.add_experiment("plan-exp")
    # Executor seam: the server task keeps PlanMode's DEFAULT executor, which
    # is constructed from the module-global LocalExecutor in harness/modes/
    # plan.py — patching that symbol makes step-7 ExecuteTests/CompileWorkflow
    # and the execute tail spawn no real subprocesses (deterministic, fast).
    monkeypatch.setattr("molexp.harness.modes.plan.LocalExecutor", DryRunExecutor)
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: workspace
    plan_gateway.set_plan_gateway_factory(_stub_factory)
    with TestClient(app) as client:  # context manager → lifespan cancels tasks on teardown
        yield client
    plan_gateway.reset_plan_gateway_factory()


_BASE = "/api/projects/test-project/experiments/plan-exp/plan-tasks"
_PLANS = "/api/projects/test-project/experiments/plan-exp/plans"


def _await_terminal(client: TestClient, task_id: str, tries: int = 400) -> dict[str, object]:
    """Poll the task until it leaves ``running`` (~20s budget, same as routes tests)."""
    url = f"{_BASE}/{task_id}"
    body: dict[str, object] = client.get(url).json()
    for _ in range(tries):
        if body["status"] != "running":
            return body
        time.sleep(0.05)
        body = client.get(url).json()
    return body


def _drive_through_approvals(
    client: TestClient, task_id: str, *, rounds: int = 8
) -> tuple[dict[str, object], list[str]]:
    """Grant every gate via the approvals inbox; return (final body, granted ids).

    The vision-loop-01 wall: server tasks never auto-grant — the HTTP path to
    a completed pipeline IS suspend → decide (inbox) → resume, once per gate.
    The granted request ids record WHICH gates the task actually rode.
    """
    granted: list[str] = []
    body = _await_terminal(client, task_id)
    for _ in range(rounds):
        if body["status"] != "waiting_approval":
            return body, granted
        inbox = client.get("/api/approvals").json()
        item = next(i for i in inbox["items"] if i["taskId"] == task_id)
        resp = client.post(
            f"/api/approvals/plan/{task_id}/decisions",
            json={"requestId": item["requestId"], "granted": True},
        )
        assert resp.status_code == 200, resp.text
        granted.append(item["requestId"])
        body = _await_terminal(client, task_id)
    return body, granted


def _as_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict), f"expected a JSON object, got {type(value)!r}"
    return value


# ──────────────────────────────────────────────────── basics: request/response


def test_execute_defaults_false_in_task_response(plan_client: TestClient) -> None:
    """category: basics — omitting ``execute`` yields ``execute: false``."""
    resp = plan_client.post(
        _BASE, json={"draft": "default execute draft", "model": "stub-model", "ground": False}
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["execute"] is False


def test_execute_true_is_reflected_in_task_response(plan_client: TestClient) -> None:
    """category: basics — ``execute: true`` echoes on POST and GET (UI label)."""
    resp = plan_client.post(
        _BASE,
        json={
            "draft": "execute-flagged draft",
            "model": "stub-model",
            "ground": False,
            "execute": True,
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["execute"] is True

    fetched = plan_client.get(f"{_BASE}/{body['taskId']}").json()
    assert fetched["execute"] is True


def test_known_compute_target_name_is_accepted(
    plan_client: TestClient, workspace: Workspace
) -> None:
    """category: basics — a registered target name passes route validation."""
    add_target(workspace, ComputeTarget(name="hpc", scratch_root="/scratch/hpc"))

    resp = plan_client.post(
        _BASE,
        json={
            "draft": "known target draft",
            "model": "stub-model",
            "ground": False,
            "execute": True,
            "compute_target": "hpc",
        },
    )
    assert resp.status_code == 201, resp.text


# ─────────────────────────────────────────────────────────── edge cases


def test_unknown_compute_target_returns_422_listing_known_names(
    plan_client: TestClient, workspace: Workspace
) -> None:
    """category: edge cases — unknown name fails fast, detail names candidates."""
    add_target(workspace, ComputeTarget(name="hpc", scratch_root="/scratch/hpc"))

    resp = plan_client.post(
        _BASE,
        json={
            "draft": "unknown target draft",
            "model": "stub-model",
            "ground": False,
            "execute": True,
            "compute_target": "no-such-target",
        },
    )
    assert resp.status_code == 422, resp.text
    assert "hpc" in resp.text, "422 detail must list the known target names"

    # Fail-fast means fail-clean: no background task was started.
    listed = plan_client.get(_BASE).json()
    assert listed["total"] == 0


# ──────────────────────────────────────── threading into PlanMode (captured)


def test_execute_and_resolved_target_thread_into_plan_mode(
    plan_client: TestClient, workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """category: integration — the driven PlanMode receives the route's flags.

    Captures the ``PlanMode(...)`` constructor kwargs the background task uses
    (the least brittle direct proof that ``execute=True`` and the *resolved*
    ``ComputeTarget`` — not the raw name — reach the harness).
    """
    import molexp.harness
    import molexp.harness.modes
    import molexp.harness.modes.plan
    from molexp.harness.modes.plan import PlanMode as RealPlanMode

    add_target(workspace, ComputeTarget(name="hpc", scratch_root="/scratch/hpc"))
    captured: list[dict[str, object]] = []

    class RecordingPlanMode(RealPlanMode):
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured.append(dict(kwargs))
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    # Cover every plausible import site of the constructor.
    monkeypatch.setattr(molexp.harness, "PlanMode", RecordingPlanMode)
    monkeypatch.setattr(molexp.harness.modes, "PlanMode", RecordingPlanMode)
    monkeypatch.setattr(molexp.harness.modes.plan, "PlanMode", RecordingPlanMode)

    resp = plan_client.post(
        _BASE,
        json={
            "draft": "capture plan-mode kwargs",
            "model": "stub-model",
            "ground": False,
            "execute": True,
            "compute_target": "hpc",
        },
    )
    assert resp.status_code == 201, resp.text

    for _ in range(400):  # PlanMode is built before the first stage — quick
        if captured:
            break
        time.sleep(0.05)
    assert captured, "the background plan task never constructed a PlanMode"

    kwargs = captured[0]
    assert kwargs.get("execute") is True
    target = kwargs.get("compute_target")
    assert isinstance(target, ComputeTarget)
    assert target.name == "hpc"


# ─────────────────────────────── lifecycle: the gated execute tail end-to-end


@pytest.mark.integration
def test_execute_false_plan_completes_without_tail_artifacts(plan_client: TestClient) -> None:
    """category: lifecycle — default flow ends at the execution report.

    Two gates only (spec + plan review), never ``approve-execution``; the plan
    detail carries no ``finalReport`` / ``auditReport`` (north-star §2.2: the
    default never submits).
    """
    resp = plan_client.post(
        _BASE, json={"draft": "plan-only water nemd", "model": "stub-model", "ground": False}
    )
    assert resp.status_code == 201, resp.text
    started = resp.json()

    final, granted = _drive_through_approvals(plan_client, started["taskId"])
    assert final["status"] == "completed", final
    assert granted == [_SPEC_GATE, _PLAN_GATE]

    detail = plan_client.get(f"{_PLANS}/{started['runId']}")
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["finalReport"] is None
    assert body["auditReport"] is None


@pytest.mark.integration
def test_execute_tail_rides_three_gates_and_surfaces_final_and_audit_reports(
    plan_client: TestClient,
) -> None:
    """category: lifecycle — execute=true suspends at ALL three inbox gates.

    Each decision comes from ``POST /api/approvals`` (slice-01 inbox, never
    auto-granted); after the third grant the tail completes and the plan
    detail route carries the real ``finalReport`` + ``auditReport``.
    """
    resp = plan_client.post(
        _BASE,
        json={
            "draft": "execute water nemd for real",
            "model": "stub-model",
            "ground": False,
            "execute": True,
        },
    )
    assert resp.status_code == 201, resp.text
    started = resp.json()
    assert started["execute"] is True

    final, granted = _drive_through_approvals(plan_client, started["taskId"])
    assert final["status"] == "completed", final
    assert granted == [_SPEC_GATE, _PLAN_GATE, _EXECUTION_GATE]
    assert final["workflowPersisted"] is True

    detail = plan_client.get(f"{_PLANS}/{started['runId']}")
    assert detail.status_code == 200, detail.text
    body = detail.json()
    final_report = _as_dict(body["finalReport"])
    assert final_report["title"] == _FINAL_REPORT["title"]
    assert body["auditReport"] is not None
