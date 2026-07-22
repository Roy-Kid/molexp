"""Regression: scope-tagged plan suspend/resume (spec plan-emergent-07-suspend-resume).

Binding runtime example (ac-009) for spec ``plan-emergent-07-suspend-resume``. A
library-external user drives durable, scope-routed PlanMode resume through the
public :mod:`molexp.services.plan_runtime` API, with the ``ResumeDriver`` seam
filled by an injected offline ``FakeResumeDriver`` — NO real router / LLM, NO
subprocess, NO TestClient / app boot. It mirrors the construction in
``tests/test_services/test_plan_resume_scope.py``.

plan-emergent-07 splits plan suspension into two resume scopes; an answered
``ApprovalRequest`` routes by its ``scope``:

1. ``approval_gate`` -> **main**: a phase-1 pipeline gate. ``resolve_resume_scope``
   returns ``"approval_gate"``; :func:`decide_plan_review` re-drives the *main*
   planning session (ledger re-drive: ``PlanTask.resume``). Route
   ``"approval_gate->main"``.
2. ``intervention_request`` -> **subagent**: a phase-2 named codegen subagent is
   stuck and asks for human guidance. ``resolve_resume_scope`` returns
   ``"intervention_request"``; :func:`decide_plan_review` threads the operator
   guidance into ``PlanTask.resume_intervention`` -> ``ResumeDriver.resume_subagent``
   for the named ``target_agent_id``. Route
   ``"intervention_request->subagent:codegen-task-T"``.

Asymmetry (documented on purpose): scope-1's *real* ``PlanTask.resume`` re-drives
PlanMode through a gateway (needs a router/LLM — not offline), so a typed
``FakeTask`` double observes the route without running it; scope-2's real
``PlanTask.resume_intervention`` only touches the injected ``ResumeDriver``, so a
*real* ``PlanTask`` drives it end to end into the fake. Both routes are reached
through the same public services entry (:func:`decide_plan_review`).

Also demonstrated: the ``SQLiteApprovalStore`` round-trip — ``record_pending`` an
intervention request, ``pending()`` reads it back with its ``scope`` and
``target_agent_id`` intact.

Deterministic + offline: fixed timestamps, no network, no subprocess, no server,
no CLI, no real LLM. Run standalone with
``python regressions/plan-emergent-07-suspend-resume.py``; the final line on
success is ``plan-emergent-07-suspend-resume: ok``.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from molexp.harness import SQLiteApprovalStore
from molexp.harness.schemas import ApprovalRequest, ApprovalScope, ReviewDecision
from molexp.services.plan_runtime import (
    ResumeDriver,
    decide_plan_review,
    reset_resume_driver_factory,
    resolve_resume_scope,
    set_resume_driver_factory,
)
from molexp.services.plan_runtime.task import PlanTask
from molexp.workspace import Experiment, Run, Workspace

#: The ordered resume routes this regression pins (the reference value).
REFERENCE_ROUTES: list[str] = [
    "approval_gate->main",
    "intervention_request->subagent:codegen-task-T",
]

#: The one stuck codegen subagent the phase-2 intervention re-seeds.
TARGET_AGENT_ID = "codegen-task-T"


class FakeResumeDriver:
    """Offline ``ResumeDriver`` double recording each resume call verbatim.

    Structurally satisfies the runtime ``ResumeDriver`` Protocol (both methods
    async); ``run`` is typed ``object`` so a workspace ``Run`` is accepted
    contravariantly without an ``Any`` escape. Mirrors the fake in
    ``tests/test_services/test_plan_resume_scope.py``.
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


class FakeTask:
    """Typed plan-task double: records which resume route ``decide_plan_review`` fired.

    Stands in for a real ``PlanTask`` on the scope-1 ``approval_gate`` route,
    whose real ``resume`` would re-drive PlanMode through a gateway/LLM (not
    offline). Only the surface ``decide_plan_review`` touches is implemented.
    """

    def __init__(self) -> None:
        self.status: str = "waiting_approval"
        self.resume_calls: int = 0
        self.intervention_calls: list[dict[str, object]] = []

    def resume(self) -> None:
        self.resume_calls += 1

    def resume_intervention(
        self, *, target_agent_id: str | None, payload: dict[str, object]
    ) -> None:
        self.intervention_calls.append({"target_agent_id": target_agent_id, "payload": payload})

    def mark_rejected(self, reason: str) -> None:  # pragma: no cover - unused route
        raise AssertionError(f"approval_gate approve must not reject: {reason!r}")


def _decided_at() -> datetime:
    """A fixed decision timestamp (determinism: no wall-clock)."""
    return datetime(2026, 7, 22, 12, 0, 0, tzinfo=UTC)


def _gate_request() -> ApprovalRequest:
    """A phase-1 ``approval_gate`` request (default scope)."""
    return ApprovalRequest(
        id="req-gate",
        intent="experiment_spec",
        reason="approve the concrete spec",
        triggered_by_policy="PlanMode",
        created_at=_decided_at(),
    )


def _intervention_request() -> ApprovalRequest:
    """A phase-2 ``intervention_request`` naming the stuck codegen subagent."""
    return ApprovalRequest(
        id="req-int",
        intent="task_intervention",
        reason="loop stuck - need human guidance",
        triggered_by_policy="StepAuditLoop",
        created_at=_decided_at(),
        scope="intervention_request",
        target_agent_id=TARGET_AGENT_ID,
    )


def _approve_review() -> ReviewDecision:
    """An ``approve`` review answering the phase-1 gate."""
    return ReviewDecision(
        pack_id="pack-1",
        action="approve",
        decided_by="ui-operator",
        decided_at=_decided_at(),
    )


def _guidance_review() -> ReviewDecision:
    """A ``revise`` review carrying operator guidance for the stuck subagent."""
    return ReviewDecision(
        pack_id="pack-1",
        action="revise",
        decided_by="ui-operator",
        decided_at=_decided_at(),
        reason="use PME electrostatics",
        field_values={"solver": "pme"},
        edits={"box": 3.0},
    )


def _build_run(root: Path) -> tuple[Experiment, Run]:
    """A real on-disk workspace/experiment/run (mirrors the unit test's ``_run``)."""
    ws = Workspace(root / "lab", name="lab")
    ws.materialize()
    experiment = ws.add_project("p").add_experiment("e")
    run = experiment.add_run(id="r1")
    return experiment, run


def _scope_1_approval_gate_to_main(run: Run) -> str:
    """Route an ``approval_gate`` decision to the main-session ledger re-drive.

    Mirrors ``test_approval_gate_keeps_existing_resume_path``: ``decide_plan_review``
    resolves the ``approval_gate`` scope to ``PlanTask.resume`` (main-conversation
    ledger re-drive), never the intervention path. A ``FakeTask`` observes it.
    """
    request = _gate_request()

    scope: ApprovalScope = resolve_resume_scope(request)
    print(f"[approval_gate] resolve_resume_scope        = {scope!r}")
    assert scope == "approval_gate", f"gate request must resolve to approval_gate, got {scope!r}"

    task = FakeTask()
    decide_plan_review(run=run, request=request, decision=_approve_review(), task=task)

    print(f"[approval_gate] PlanTask.resume calls        = {task.resume_calls}")
    print(f"[approval_gate] PlanTask.resume_intervention = {len(task.intervention_calls)}")
    assert task.resume_calls == 1, (
        f"approval_gate approve must re-drive the main session once, got {task.resume_calls}"
    )
    assert task.intervention_calls == [], (
        "approval_gate must never route into the subagent-intervention path"
    )
    return "approval_gate->main"


def _store_round_trip(store: SQLiteApprovalStore, run: Run) -> None:
    """Persist an intervention request and read it back with scope + target intact."""
    request = _intervention_request()
    store.record_pending(run.id, request)
    pending = store.pending(run.id)

    print(f"[store] pending count                   = {len(pending)}")
    assert len(pending) == 1, f"exactly one pending request expected, got {len(pending)}"
    read_back = pending[0]
    print(f"[store] pending[0].scope                = {read_back.scope!r}")
    print(f"[store] pending[0].target_agent_id      = {read_back.target_agent_id!r}")
    assert read_back.id == "req-int", f"round-trip id mismatch: {read_back.id!r}"
    assert read_back.scope == "intervention_request", (
        f"round-trip must preserve scope, got {read_back.scope!r}"
    )
    assert read_back.target_agent_id == TARGET_AGENT_ID, (
        f"round-trip must preserve target_agent_id, got {read_back.target_agent_id!r}"
    )


async def _scope_2_intervention_to_subagent(
    experiment: Experiment, run: Run, fake: FakeResumeDriver
) -> str:
    """Route an ``intervention_request`` decision into the named codegen subagent.

    Mirrors ``test_intervention_threads_guidance_into_resume_intervention`` +
    ``test_resume_intervention_routes_to_subagent``: ``decide_plan_review`` threads
    the operator guidance into the *real* ``PlanTask.resume_intervention``, which
    re-seeds the named subagent through ``ResumeDriver.resume_subagent`` (the
    injected ``fake``) — never the main session.
    """
    request = _intervention_request()

    scope: ApprovalScope = resolve_resume_scope(request)
    print(f"[intervention] resolve_resume_scope     = {scope!r}")
    assert scope == "intervention_request", (
        f"intervention request must resolve to intervention_request, got {scope!r}"
    )

    task = PlanTask(
        task_id="t1",
        run=run,
        experiment=experiment,
        draft="draft",
        model="fake-model",
        created_at="2026-07-22T00:00:00Z",
    )
    task.status = "waiting_approval"

    decide_plan_review(run=run, request=request, decision=_guidance_review(), task=task)
    await task.await_finished()

    print(f"[intervention] resume_subagent calls    = {len(fake.subagent_calls)}")
    print(f"[intervention] resume_main calls        = {len(fake.main_calls)}")
    assert len(fake.subagent_calls) == 1, (
        f"exactly one subagent re-seed expected, got {len(fake.subagent_calls)}"
    )
    assert fake.main_calls == [], "an intervention must never re-drive the main session"

    call = fake.subagent_calls[0]
    print(f"[intervention] target_agent_id          = {call['target_agent_id']!r}")
    print(f"[intervention] payload                  = {call['payload']!r}")
    assert call["target_agent_id"] == TARGET_AGENT_ID, (
        f"subagent re-seed must name the stuck agent, got {call['target_agent_id']!r}"
    )
    assert call["run"] is run, "subagent re-seed must carry the suspended run verbatim"
    payload = call["payload"]
    assert isinstance(payload, dict), f"payload must be a mapping, got {type(payload)!r}"
    assert payload["field_values"] == {"solver": "pme"}, (
        f"guidance field_values must be delivered verbatim, got {payload['field_values']!r}"
    )
    assert payload["reason"] == "use PME electrostatics", (
        f"guidance reason must be delivered verbatim, got {payload['reason']!r}"
    )
    assert payload["edits"] == {"box": 3.0}, (
        f"guidance edits must be delivered verbatim, got {payload['edits']!r}"
    )
    return f"intervention_request->subagent:{TARGET_AGENT_ID}"


async def main() -> int:
    """Drive both resume scopes end to end; print the ordered routes + success marker."""
    tmp = Path(tempfile.mkdtemp(prefix="plan-emergent-07-suspend-resume-"))
    fake = FakeResumeDriver()
    set_resume_driver_factory(lambda *_args, **_kwargs: fake)
    try:
        # The injected fake structurally satisfies the resume seam
        # (mirrors test_fake_driver_satisfies_runtime_protocol).
        print(f"[seam] FakeResumeDriver is ResumeDriver = {isinstance(fake, ResumeDriver)}")
        assert isinstance(fake, ResumeDriver), (
            "FakeResumeDriver must structurally satisfy the ResumeDriver Protocol"
        )

        experiment, run = _build_run(tmp)
        observed_routes: list[str] = []

        print("== scope 1: approval_gate -> main (ledger re-drive) ==")
        observed_routes.append(_scope_1_approval_gate_to_main(run))

        print("== store: intervention_request round-trips scope + target_agent_id ==")
        _store_round_trip(SQLiteApprovalStore(path=tmp / "approvals.sqlite"), run)

        print("== scope 2: intervention_request -> subagent:codegen-task-T ==")
        observed_routes.append(await _scope_2_intervention_to_subagent(experiment, run, fake))

        print(f"[routes] reference = {REFERENCE_ROUTES}")
        print(f"[routes] observed  = {observed_routes}")
        assert observed_routes == REFERENCE_ROUTES, (
            f"resolved routes must match the reference, got {observed_routes}"
        )
    finally:
        reset_resume_driver_factory()
        shutil.rmtree(tmp, ignore_errors=True)

    print("plan-emergent-07-suspend-resume: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
