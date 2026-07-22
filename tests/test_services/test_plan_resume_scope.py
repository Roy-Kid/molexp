"""Scope-routed durable resume (spec plan-emergent-07).

plan-emergent-07 splits plan suspension into two scopes and routes resume by
the answered request's ``scope``:

* ``approval_gate`` (phase-1) — the existing ledger re-drive via
  ``PlanTask.resume`` / ``decide_plan_review`` (unchanged).
* ``intervention_request`` (phase-2) — a named codegen subagent is re-seeded
  and re-run through the ``ResumeDriver`` seam
  (``PlanTask.resume_intervention`` -> ``ResumeDriver.resume_subagent``).

All resume driving is offline-deterministic here: a ``FakeResumeDriver`` is
injected through ``set_resume_driver_factory`` (mirroring the existing
``set_plan_gateway_factory`` seed), so no real router / LLM / subprocess runs.
The ac-007 inbox surfacing is covered at the *function* level (build the
``PendingApprovalItem`` mapping directly, no TestClient / app boot).

RED before GREEN: pins the ``resume_scope`` module + the generalized
``decide_plan_review`` / ``PlanTask.resume_intervention`` the production change
must satisfy.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from molexp.harness.schemas import ApprovalRequest, ReviewDecision
from molexp.services.plan_runtime import resume_scope
from molexp.services.plan_runtime.decide import decide_plan_review
from molexp.services.plan_runtime.resume_scope import (
    ResumeDriver,
    propose_plan_patch,
    reset_resume_driver_factory,
    resolve_resume_scope,
    set_resume_driver_factory,
)
from molexp.services.plan_runtime.task import PlanTask
from molexp.workspace import Workspace


class FakeResumeDriver:
    """Offline ``ResumeDriver`` double recording each resume call verbatim.

    Structurally satisfies the ``ResumeDriver`` runtime Protocol (both methods
    async); ``run`` typed as ``object`` so a workspace ``Run`` is accepted
    contravariantly without an ``Any`` escape.
    """

    def __init__(self) -> None:
        self.main_calls: list[dict[str, object]] = []
        self.subagent_calls: list[dict[str, object]] = []

    async def resume_main(self, *, run: object, payload: dict[str, object]) -> None:
        self.main_calls.append({"run": run, "payload": payload})

    async def resume_subagent(
        self, *, run: object, target_agent_id: str, payload: dict[str, object]
    ) -> None:
        self.subagent_calls.append(
            {"run": run, "target_agent_id": target_agent_id, "payload": payload}
        )


@pytest.fixture(autouse=True)
def _reset_driver_factory() -> Iterator[None]:
    """Drop any seeded resume-driver factory after each test (seam hygiene)."""
    yield
    reset_resume_driver_factory()


def _run(tmp_path: Path) -> tuple[object, object]:
    ws = Workspace(tmp_path / "lab", name="lab")
    ws.materialize()
    experiment = ws.add_project("p").add_experiment("e")
    run = experiment.add_run(id="r1")
    return experiment, run


def _decided_at() -> datetime:
    return datetime(2026, 7, 22, 12, 0, 0, tzinfo=UTC)


def _gate_request() -> ApprovalRequest:
    return ApprovalRequest(
        id="req-gate",
        intent="experiment_spec",
        reason="approve the concrete spec",
        triggered_by_policy="PlanMode",
        created_at=_decided_at(),
    )


def _intervention_request(target: str | None = "codegen-task-T") -> ApprovalRequest:
    return ApprovalRequest(
        id="req-int",
        intent="task_intervention",
        reason="loop stuck — need human guidance",
        triggered_by_policy="StepAuditLoop",
        created_at=_decided_at(),
        scope="intervention_request",
        target_agent_id=target,
    )


def _review(
    action: str = "revise",
    *,
    field_values: dict[str, object] | None = None,
    reason: str | None = None,
    edits: dict[str, object] | None = None,
) -> ReviewDecision:
    return ReviewDecision(
        pack_id="pack-1",
        action=action,  # type: ignore[arg-type]
        decided_by="ui-operator",
        decided_at=_decided_at(),
        reason=reason,
        field_values=field_values or {},
        edits=edits,
    )


class TestResolveResumeScope:
    def test_returns_approval_gate_scope(self) -> None:
        assert resolve_resume_scope(_gate_request()) == "approval_gate"

    def test_returns_intervention_scope(self) -> None:
        assert resolve_resume_scope(_intervention_request()) == "intervention_request"


class TestResumeDriverSeam:
    def test_fake_driver_satisfies_runtime_protocol(self) -> None:
        assert isinstance(FakeResumeDriver(), ResumeDriver)

    @pytest.mark.asyncio
    async def test_resume_intervention_routes_to_subagent(self, tmp_path: Path) -> None:
        experiment, run = _run(tmp_path)
        task = PlanTask(
            task_id="t1",
            run=run,  # type: ignore[arg-type]
            experiment=experiment,  # type: ignore[arg-type]
            draft="d",
            model="m",
            created_at="2026-07-22T00:00:00Z",
        )
        task.status = "waiting_approval"
        fake = FakeResumeDriver()
        set_resume_driver_factory(lambda *args, **kwargs: fake)

        task.resume_intervention(target_agent_id="codegen-task-T", payload={"reason": "tweak"})
        assert task._task is not None
        with contextlib.suppress(Exception):
            await task._task

        assert len(fake.subagent_calls) == 1
        call = fake.subagent_calls[0]
        assert call["target_agent_id"] == "codegen-task-T"
        assert call["payload"] == {"reason": "tweak"}
        assert call["run"] is run
        assert fake.main_calls == []

    def test_resume_intervention_without_target_raises(self, tmp_path: Path) -> None:
        experiment, run = _run(tmp_path)
        task = PlanTask(
            task_id="t1",
            run=run,  # type: ignore[arg-type]
            experiment=experiment,  # type: ignore[arg-type]
            draft="d",
            model="m",
            created_at="2026-07-22T00:00:00Z",
        )
        task.status = "waiting_approval"
        set_resume_driver_factory(lambda *args, **kwargs: FakeResumeDriver())

        # No silent fallback: an intervention resume with no named subagent fails loud.
        with pytest.raises((ValueError, RuntimeError), match="target"):
            task.resume_intervention(target_agent_id=None, payload={})


class TestProposePlanPatchFallback:
    @pytest.mark.asyncio
    async def test_propose_plan_patch_reopens_main_conversation(self, tmp_path: Path) -> None:
        _experiment, run = _run(tmp_path)
        fake = FakeResumeDriver()
        set_resume_driver_factory(lambda *args, **kwargs: fake)

        awaitable = propose_plan_patch(run=run, patch={"note": "flip solver to PME"})
        if awaitable is not None:
            await awaitable
        # tolerate a sync-spawning impl (awaitable is None): let its task run.
        import asyncio

        await asyncio.sleep(0)

        assert len(fake.main_calls) == 1
        call = fake.main_calls[0]
        assert call["run"] is run
        assert "flip solver to PME" in json.dumps(call["payload"], default=str)
        assert fake.subagent_calls == []


class TestDecidePlanReviewScopeRouting:
    def test_approval_gate_keeps_existing_resume_path(self, tmp_path: Path) -> None:
        _experiment, run = _run(tmp_path)
        task = MagicMock()
        task.status = "waiting_approval"
        decide_plan_review(
            run=run,  # type: ignore[arg-type]
            request=_gate_request(),
            decision=_review("approve"),
            task=task,
        )
        task.resume.assert_called_once()
        task.resume_intervention.assert_not_called()

    def test_intervention_threads_guidance_into_resume_intervention(self, tmp_path: Path) -> None:
        _experiment, run = _run(tmp_path)
        task = MagicMock()
        task.status = "waiting_approval"
        decide_plan_review(
            run=run,  # type: ignore[arg-type]
            request=_intervention_request(),
            decision=_review(
                field_values={"solver": "pme"},
                reason="use PME electrostatics",
                edits={"box": 3.0},
            ),
            task=task,
        )
        task.resume_intervention.assert_called_once()
        _args, kwargs = task.resume_intervention.call_args
        assert kwargs["target_agent_id"] == "codegen-task-T"
        payload = kwargs["payload"]
        assert payload["field_values"] == {"solver": "pme"}
        assert payload["reason"] == "use PME electrostatics"
        assert payload["edits"] == {"box": 3.0}
        task.resume.assert_not_called()

    def test_plan_patch_fallback_invokes_propose_plan_patch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _experiment, run = _run(tmp_path)
        task = MagicMock()
        task.status = "waiting_approval"
        recorder = MagicMock()
        # decide looks up propose_plan_patch via the resume_scope module (lazy).
        monkeypatch.setattr(resume_scope, "propose_plan_patch", recorder)

        decide_plan_review(
            run=run,  # type: ignore[arg-type]
            request=_intervention_request(),
            decision=_review(field_values={"fallback": "plan_patch", "solver": "pme"}),
            task=task,
        )
        recorder.assert_called_once()
        _args, kwargs = recorder.call_args
        assert kwargs["run"] is run
        task.resume_intervention.assert_not_called()

    def test_intervention_without_target_raises(self, tmp_path: Path) -> None:
        _experiment, run = _run(tmp_path)
        task = MagicMock()
        task.status = "waiting_approval"
        with pytest.raises((ValueError, RuntimeError), match="target"):
            decide_plan_review(
                run=run,  # type: ignore[arg-type]
                request=_intervention_request(target=None),
                decision=_review(field_values={}),
                task=task,
            )
        task.resume_intervention.assert_not_called()


# ---- ac-007: inbox item mapping (function level, no TestClient / app boot) ----


class _FakeProject:
    id = "proj-x"


class _FakeExperiment:
    id = "exp-x"
    project = _FakeProject()


class _FakeInboxTask:
    """Minimal duck-typed plan task for the inbox item-mapping function."""

    task_id = "task-x"
    run_id = "run-x"
    run = object()
    experiment = _FakeExperiment()

    def __init__(self, requests: list[ApprovalRequest]) -> None:
        self.pending_requests = requests


class TestApprovalInboxItemMapping:
    def test_intervention_item_surfaces_scope_and_target_agent(self) -> None:
        # _items_for is the extractable mapping in routes/approvals.py; it turns
        # each task's pending ApprovalRequests into PendingApprovalItems without
        # any app boot. Called directly here to pin scope/targetAgentId surfacing.
        from molexp.server.routes.approvals import _items_for

        items = _items_for("plan", [_FakeInboxTask([_intervention_request()])])

        assert len(items) == 1
        item = items[0]
        assert item.requestId == "req-int"
        assert item.intent == "task_intervention"
        assert item.scope == "intervention_request"
        assert item.targetAgentId == "codegen-task-T"
