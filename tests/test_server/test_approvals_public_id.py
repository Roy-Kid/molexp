"""Approvals inbox uses the single public conversation id as taskId."""

from __future__ import annotations

from datetime import UTC, datetime

from molexp.harness.schemas import ApprovalRequest
from molexp.server.routes.approvals import _items_for, _public_task_id


class _FakeProject:
    id = "proj"


class _FakeExperiment:
    id = "exp"
    project = _FakeProject()


class _FakePlanTask:
    def __init__(self, *, task_id: str, record_task_id: str | None, run_id: str = "run-1") -> None:
        self.task_id = task_id
        self.record_task_id = record_task_id
        self.run_id = run_id
        self.run = object()
        self.experiment = _FakeExperiment()
        self.pending_requests = [
            ApprovalRequest(
                id="req-1",
                intent="approve_experiment_plan",
                reason="review",
                triggered_by_policy="hard",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ]


def test_public_task_id_prefers_record_task_id() -> None:
    t = _FakePlanTask(task_id="plan-internal", record_task_id="task-agent")
    assert _public_task_id(t) == "task-agent"
    assert _public_task_id(_FakePlanTask(task_id="plan-x", record_task_id=None)) == "plan-x"


def test_items_for_emits_agent_task_id() -> None:
    t = _FakePlanTask(task_id="plan-internal", record_task_id="task-agent")
    items = _items_for("plan", [t])
    assert len(items) == 1
    assert items[0].taskId == "task-agent"
    assert items[0].runId == "run-1"
    assert items[0].projectId == "proj"


def test_registry_get_resolves_unified_id(tmp_path) -> None:
    from molexp.services.plan_runtime.registry import PlanTaskRegistry

    # Smoke: get by missing key returns None without crash.
    reg = PlanTaskRegistry()
    assert reg.get(str(tmp_path), "task-missing") is None
