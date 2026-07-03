"""One pending-approvals inbox over HTTP (spec vision-loop-01-approval-inbox).

- ``GET /api/approvals`` aggregates the pending requests of every
  ``waiting_approval`` task across BOTH gate families — a suspended plan task
  and a suspended curate task share one inbox.
- ``POST /api/approvals/{task_kind}/{task_id}/decisions`` persists the
  ``ApprovalDecision`` (``decided_by="ui-operator"``) into that run's approval
  store and resumes the task; 404 on unknown task/request, 409 when the task
  is not ``waiting_approval``.
- ``GET /api/approvals/events`` is an SSE stream that emits whenever a task
  suspends or a decision lands (payload-free ``changed`` ping; the UI
  refetches the list).

Offline: the plan gateway factory cans every PlanMode agent; the curate
gateway plans a destructive ``move_run`` (→ §8 ChangeProposal gate → pending
with no approver); the curation-registry seam is stubbed so molmcp never
spawns.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from molexp.harness.capabilities import curation_capabilities
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.registry import InMemoryCapabilityRegistry
from molexp.harness.store.approval_store import SQLiteApprovalStore
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.server.app import create_app
from molexp.server.dependencies import get_workspace
from molexp.services.curate_runtime import gateway as curate_gateway
from molexp.services.plan_runtime import gateway as plan_gateway

_PLAN_BASE = "/api/projects/test-project/experiments/plan-exp/plan-tasks"
_CURATE_BASE = "/api/projects/test-project/experiments/curate-exp/curate-tasks"
_INBOX = "/api/approvals"

# ── canned plan-stage outputs (mirrors tests/test_server/test_plan_tasks_routes.py) ──

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
    "resource_policy": {"backend": "local", "max_runtime_s": 3600, "denied_paths": ["/", "~/.ssh"]},
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


def _plan_stub_factory(run: Any, model: str) -> Any:
    store = FileArtifactStore(root=Path(run.run_dir) / "artifacts")
    gw = StubAgentGateway(artifact_store=store)
    gw.register("experiment_report_writer", _EXPERIMENT_REPORT, output_kind="experiment_report")
    gw.register("experiment_spec_generator", _EXPERIMENT_SPEC, output_kind="experiment_spec")
    gw.register("workflow_ir_extractor", _WORKFLOW_IR, output_kind="workflow_ir")
    gw.register("bound_workflow_binder", _BOUND_WORKFLOW, output_kind="bound_workflow")
    gw.register("workflow_source_writer", _WORKFLOW_SOURCE, output_kind="workflow_source")
    gw.register(
        "plan_reviewer",
        {"passed": True, "findings": [], "summary": "faithful"},
        output_kind="plan_review",
    )
    gw.register("test_spec_writer", _TEST_SPEC, output_kind="test_spec")
    gw.register("test_code_writer", _TEST_SOURCE, output_kind="test_source")
    gw.register("input_set_generator", _INPUT_SET, output_kind="input_set")
    return gw


def _move_factory(run: Any, model: str) -> Any:
    """Curation planner that relocates run ``subject`` — destructive → gated."""
    gw = StubAgentGateway(FileArtifactStore(root=Path(run.run_dir) / "artifacts"))
    gw.register(
        "curation_planner",
        {
            "capability_id": "molexp.curation.move_run",
            "references": {"run": "subject", "target_experiment": "target-exp"},
            "reason": "relocate",
        },
        output_kind="curation_invocation",
    )
    return gw


@pytest.fixture(autouse=True)
def _patched_curation_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the merged-registry seam so molmcp is never spawned in route tests."""

    async def _stub_registry(workspace_root: str, **_: Any) -> InMemoryCapabilityRegistry:
        return InMemoryCapabilityRegistry(curation_capabilities())

    monkeypatch.setattr(
        "molexp.services.curate_runtime.flow.aresolve_curation_capability_registry",
        _stub_registry,
    )


@pytest.fixture
def client(workspace: Any, project: Any) -> Iterator[TestClient]:
    project.add_experiment("plan-exp")
    project.add_experiment("curate-exp")
    project.add_experiment("target-exp")
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: workspace
    plan_gateway.set_plan_gateway_factory(_plan_stub_factory)
    curate_gateway.set_curate_gateway_factory(_move_factory)
    with TestClient(app) as test_client:  # lifespan cancels bg tasks on teardown
        yield test_client
    plan_gateway.reset_plan_gateway_factory()
    curate_gateway.reset_curate_gateway_factory()


def _await_status(client: TestClient, url: str, status: str, tries: int = 200) -> dict[str, Any]:
    """Poll a task until it reaches *status* (minimal-polling budget ≈ 10 s)."""
    body: dict[str, Any] = client.get(url).json()
    for _ in range(tries):
        if body.get("status") == status:
            return body
        time.sleep(0.05)
        body = client.get(url).json()
    raise AssertionError(f"task never reached {status!r}: {body}")


def _suspend_plan_task(client: TestClient, draft: str = "screen solvent ratios") -> dict[str, Any]:
    resp = client.post(_PLAN_BASE, json={"draft": draft, "model": "stub-model", "ground": False})
    assert resp.status_code == 201, resp.text
    started: dict[str, Any] = resp.json()
    _await_status(client, f"{_PLAN_BASE}/{started['taskId']}", "waiting_approval")
    return started


def _suspend_curate_task(client: TestClient, workspace: Any) -> dict[str, Any]:
    project = workspace.get_project("test-project")
    project.get_experiment("curate-exp").add_run({"seed": 0}, id="subject")
    resp = client.post(
        _CURATE_BASE, json={"request": "move subject run to target-exp", "model": "m"}
    )
    assert resp.status_code == 201, resp.text
    started: dict[str, Any] = resp.json()
    _await_status(client, f"{_CURATE_BASE}/{started['taskId']}", "waiting_approval")
    return started


def _inbox_item(client: TestClient, task_id: str) -> dict[str, Any]:
    items = client.get(_INBOX).json()["items"]
    matching = [item for item in items if item["taskId"] == task_id]
    assert matching, f"task {task_id!r} not listed in the inbox: {items}"
    return matching[0]


def _plan_approval_store(workspace: Any, run_id: str) -> SQLiteApprovalStore:
    run = workspace.get_project("test-project").get_experiment("plan-exp").get_run(run_id)
    return SQLiteApprovalStore(Path(run.run_dir) / "harness.sqlite")


# ─────────────────────────────────────────────── one inbox, two kinds


@pytest.mark.integration
def test_inbox_lists_pending_requests_from_both_task_kinds(
    client: TestClient, workspace: Any
) -> None:
    plan_started = _suspend_plan_task(client)
    curate_started = _suspend_curate_task(client, workspace)

    resp = client.get(_INBOX)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 2

    by_kind = {item["taskKind"]: item for item in body["items"]}
    assert set(by_kind) == {"plan", "curate"}

    plan_item = by_kind["plan"]
    assert plan_item["taskId"] == plan_started["taskId"]
    assert plan_item["runId"] == plan_started["runId"]
    assert plan_item["projectId"] == "test-project"
    assert plan_item["experimentId"] == "plan-exp"
    assert plan_item["intent"] == "experiment_spec"
    assert plan_item["requestId"]
    assert plan_item["requestedAt"]

    curate_item = by_kind["curate"]
    assert curate_item["taskId"] == curate_started["taskId"]
    assert curate_item["requestId"]


# ─────────────────────────────────────────────── decide: grant / reject


@pytest.mark.integration
def test_grant_decision_persists_ui_operator_and_resumes_the_task(
    client: TestClient, workspace: Any
) -> None:
    started = _suspend_plan_task(client)
    item = _inbox_item(client, started["taskId"])

    resp = client.post(
        f"{_INBOX}/plan/{started['taskId']}/decisions",
        json={"requestId": item["requestId"], "granted": True},
    )
    assert resp.status_code == 200, resp.text

    # resume() flips the task out of waiting_approval synchronously …
    body = client.get(f"{_PLAN_BASE}/{started['taskId']}").json()
    assert body["status"] == "running"
    # … the grant is persisted with the HTTP caller as the decider …
    stored = _plan_approval_store(workspace, started["runId"]).granted_decision_for(
        item["requestId"]
    )
    assert stored is not None
    assert stored.decided_by == "ui-operator"
    # … and the request left the inbox.
    inbox = client.get(_INBOX).json()
    assert all(i["requestId"] != item["requestId"] for i in inbox["items"])


@pytest.mark.integration
def test_reject_decision_records_rejection_and_fails_the_task(
    client: TestClient, workspace: Any
) -> None:
    started = _suspend_plan_task(client)
    item = _inbox_item(client, started["taskId"])

    resp = client.post(
        f"{_INBOX}/plan/{started['taskId']}/decisions",
        json={"requestId": item["requestId"], "granted": False, "reason": "not safe"},
    )
    assert resp.status_code == 200, resp.text

    body = _await_status(client, f"{_PLAN_BASE}/{started['taskId']}", "failed", tries=40)
    assert body["status"] == "failed"
    # Recorded as decided-rejected: no replayable grant, no longer pending.
    store = _plan_approval_store(workspace, started["runId"])
    assert store.granted_decision_for(item["requestId"]) is None
    assert store.pending(started["runId"]) == []


# ─────────────────────────────────────────────── error contracts


@pytest.mark.integration
def test_decision_on_a_non_waiting_task_returns_409(client: TestClient) -> None:
    started = _suspend_plan_task(client)
    item = _inbox_item(client, started["taskId"])
    url = f"{_INBOX}/plan/{started['taskId']}/decisions"
    payload = {"requestId": item["requestId"], "granted": True}

    assert client.post(url, json=payload).status_code == 200
    # The task resumed (running) — a second decision has nothing to decide.
    assert client.post(url, json=payload).status_code == 409


def test_decision_for_an_unknown_task_returns_404(client: TestClient) -> None:
    resp = client.post(
        f"{_INBOX}/plan/task-does-not-exist/decisions",
        json={"requestId": "whatever", "granted": True},
    )
    assert resp.status_code == 404


@pytest.mark.integration
def test_decision_for_an_unknown_request_id_returns_404(client: TestClient) -> None:
    started = _suspend_plan_task(client)
    resp = client.post(
        f"{_INBOX}/plan/{started['taskId']}/decisions",
        json={"requestId": "request-does-not-exist", "granted": True},
    )
    assert resp.status_code == 404


# ─────────────────────────────────────────────── SSE: emits on suspend + decision
#
# TestClient buffers a streaming response to completion on `stream()` entry, so
# an endless SSE endpoint cannot be consumed through it (same reason
# tests/test_server/test_agent_sse_route.py drives the route directly). The
# endpoint's registration is asserted via OpenAPI; emission is asserted by
# driving the route's StreamingResponse body and the broadcast primitive that
# both the suspend path (task drivers) and the decide route notify.


async def test_broadcast_primitive_pings_subscribers_on_notify() -> None:
    """The asyncio broadcast the registries notify — one ping per change."""
    from molexp.services.approval_notify import (
        notify_approvals_changed,
        subscribe_approvals_changed,
    )

    subscription = subscribe_approvals_changed()
    ping = asyncio.ensure_future(anext(subscription))
    await asyncio.sleep(0)  # let the subscriber register before the notify

    notify_approvals_changed()

    await asyncio.wait_for(ping, timeout=1.0)
    await subscription.aclose()


def test_sse_endpoint_is_registered(client: TestClient) -> None:
    paths = client.get("/api/openapi.json").json()["paths"]
    assert f"{_INBOX}/events" in paths


def _text(chunk: Any) -> str:
    return chunk if isinstance(chunk, str) else chunk.decode()


async def test_sse_route_emits_a_changed_event_per_notification() -> None:
    """Driving the route directly: one ``changed`` SSE frame per broadcast ping
    — the same ping a task suspension and a landed decision emit."""
    from molexp.server.routes.approvals import stream_approval_events
    from molexp.services.approval_notify import notify_approvals_changed

    resp = await stream_approval_events()
    assert resp.media_type == "text/event-stream"
    assert resp.headers["cache-control"] == "no-cache"

    iterator = resp.body_iterator
    connected = await asyncio.wait_for(anext(iterator), timeout=1.0)
    assert "changed" in _text(connected)  # the connect frame

    ping = asyncio.ensure_future(anext(iterator))
    await asyncio.sleep(0.01)  # let the generator reach its subscription await
    notify_approvals_changed()

    frame = await asyncio.wait_for(ping, timeout=1.0)
    assert "changed" in _text(frame)
    await iterator.aclose()
