"""Plan-task caller honesty (vision-loop-06): ``recordErrors`` on the status
response; the failure branch materializes; an ApprovalPendingError suspension
materializes NOTHING.

``drive_plan_mode`` is stubbed at the services seam (no LLM, no pipeline) so
these tests pin the CALLER contract of ``PlanTask._drive`` + the route schema,
deterministically and fast.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import create_app
from molexp.server.dependencies import get_workspace
from molexp.services.agent_task_store import list_agent_task_metadata
from molexp.services.plan_runtime import gateway as plan_gateway
from molexp.workspace import Bundle
from molexp.workspace.knowledge_item import KnowledgeItem

_BASE = "/api/projects/test-project/experiments/test-exp/plan-tasks"
_DRIVE_SEAM = "molexp.services.plan_runtime.drive.drive_plan_mode"


@pytest.fixture()
def client(workspace: Any, experiment: Any) -> Iterator[TestClient]:
    """App over the shared workspace/experiment fixtures; stub gateway factory
    (the drive seam is stubbed per test, so the gateway is never exercised)."""
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: workspace
    plan_gateway.set_plan_gateway_factory(lambda *_a, **_kw: object())
    try:
        with TestClient(app) as tc:  # lifespan cancels bg tasks on teardown
            yield tc
    finally:
        plan_gateway.reset_plan_gateway_factory()


def _post(client: TestClient, draft: str) -> dict[str, Any]:
    resp = client.post(_BASE, json={"draft": draft, "model": "stub-model", "ground": False})
    assert resp.status_code == 201, resp.text
    return resp.json()


def _await_settled(client: TestClient, task_id: str, tries: int = 300) -> dict[str, Any]:
    """Poll the task until it leaves ``running`` (the bg loop runs concurrently)."""
    body: dict[str, Any] = client.get(f"{_BASE}/{task_id}").json()
    for _ in range(tries):
        if body["status"] != "running":
            return body
        time.sleep(0.02)
        body = client.get(f"{_BASE}/{task_id}").json()
    return body


def _await_true(condition: Callable[[], bool], tries: int = 300) -> bool:
    """Retry *condition* — record materialization is offloaded to a thread and
    may land a beat after the status flips."""
    for _ in range(tries):
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def _failure_items(workspace: Any) -> list[KnowledgeItem]:
    return [
        c
        for c in Bundle(workspace.root).walk()
        if isinstance(c, KnowledgeItem) and c.read_knowledge_meta().kind == "FailureAnalysis"
    ]


def test_record_errors_surface_on_the_status_response(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed record write lands structurally in ``recordErrors`` — visible on
    the very response the UI polls, never swallowed into a server log."""

    async def _fake_drive(*args: Any, **kwargs: Any) -> Any:
        return object()  # pipeline "succeeded"; only materialization matters here

    def _boom(**kwargs: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(_DRIVE_SEAM, _fake_drive)
    monkeypatch.setattr("molexp.services.plan_runtime.record.write_agent_task_record", _boom)

    started = _post(client, "records honesty draft")
    final = _await_settled(client, started["taskId"])

    assert final["status"] == "completed", final
    assert final["recordErrors"], "a failed record write must surface on the response"
    joined = " ".join(final["recordErrors"])
    assert "agent_task" in joined
    assert "disk full" in joined


def test_failed_plan_task_materializes_failure_records(
    client: TestClient, workspace: Any, experiment: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The server task's failure branch calls materialize(failure=...): the
    FailureAnalysis item + the failed Agents-tab entry exist on disk."""

    async def _fake_drive(*args: Any, **kwargs: Any) -> Any:
        from molexp.harness import StageExecutionError

        raise StageExecutionError("stage 'generate_experiment_spec' exploded")

    monkeypatch.setattr(_DRIVE_SEAM, _fake_drive)

    started = _post(client, "terminally failing draft")
    final = _await_settled(client, started["taskId"])

    assert final["status"] == "failed", final
    run_id = started["runId"]
    name = f"failure-{experiment.id}-{run_id}"

    def _analysis_landed() -> bool:
        # The failure-branch materialization runs on a worker thread AFTER the
        # status flips — retry the FULL read (dir + meta + body) to avoid
        # observing a half-written item.
        try:
            item = experiment.get_folder(name, cls=KnowledgeItem)
            return (
                item.read_knowledge_meta().kind == "FailureAnalysis" and "exploded" in item.body()
            )
        except Exception:
            return False

    assert _await_true(_analysis_landed), (
        "a terminally failed plan task must materialize a FailureAnalysis"
    )
    assert _await_true(
        lambda: any(
            t.task_id == started["taskId"] and t.status == "failed"
            for t in list_agent_task_metadata(str(workspace.root))
        )
    ), "a failed plan must show in the Agents tab with status failed"


def test_approval_suspension_materializes_no_failure_records(
    client: TestClient, workspace: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ApprovalPendingError is a suspension, not a failure: the task lands
    ``waiting_approval`` and NEITHER a FailureAnalysis NOR a failed agent-task
    entry is written — a plan that later resumes and succeeds must not leave a
    phantom failure in the knowledge base."""

    async def _fake_drive(*args: Any, **kwargs: Any) -> Any:
        from molexp.harness import ApprovalPendingError
        from molexp.harness.schemas import ApprovalRequest

        request = ApprovalRequest(
            id="req-1",
            intent="experiment_spec",
            reason="human gate",
            triggered_by_policy="test",
            created_at=datetime.now(tz=UTC),
        )
        raise ApprovalPendingError([request], kwargs["run"].id)

    monkeypatch.setattr(_DRIVE_SEAM, _fake_drive)

    started = _post(client, "suspending draft")
    final = _await_settled(client, started["taskId"])

    assert final["status"] == "waiting_approval", final
    assert final["recordErrors"] == []
    time.sleep(0.1)  # give any (wrong) stray materialization a beat to land
    assert _failure_items(workspace) == [], "a suspension must not materialize a FailureAnalysis"
    # The task IS visible in the Agents hub while suspended — with its honest
    # waiting_approval status, never a phantom "failed".
    entry = next(
        (
            t
            for t in list_agent_task_metadata(str(workspace.root))
            if t.task_id == started["taskId"]
        ),
        None,
    )
    assert entry is not None, "a suspended plan task must be listed in the agent-task store"
    assert entry.status == "waiting_approval", entry
    assert entry.plan_mode is True
