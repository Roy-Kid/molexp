"""Regression: EmergentPlanOrchestrator end-to-end, offline (plan-emergent-05c-orchestrator).

Binding runtime example for spec ``plan-emergent-05c-orchestrator``. A
library-external user drives the **public** emergent-planning orchestrator
(:class:`~molexp.harness.modes.emergent_plan.EmergentPlanOrchestrator`) through
its documented run-shape, injecting a SELF-CONTAINED stub ``PlanLoopRunner``
(writes a canned board via ``write_board`` — no real Pi loop) and a
:class:`~molexp.harness.gateways.stub.StubAgentGateway` carrying a canned
``plan_report_renderer`` response. No network, no subprocess, no server, no CLI,
no LLM; deterministic on every run (the freeze is content-addressed and the
artifact store / approval store are on-disk).

Three reference observations reproduce the orchestrator's binding behaviors,
each printed before it is asserted:

(a) **guard steer-back / no human surface** — a persistently MALFORMED final
    board (a task with no acceptance) is rejected by the deterministic
    ``EmergentPlanFormValidator`` guard BEFORE the human gate: no ``review_pack``
    and no ``frozen_experiment_plan`` artifact are written, and the approval
    store has NO pending request. A malformed plan never reaches the human.
(b) **store-first suspend** — a VALID board with NO approver and NO stored grant
    suspends store-first: ``run(...)`` raises :class:`ApprovalPendingError` and
    ``SQLiteApprovalStore(...).pending(run.id)`` lists the pending
    ``approve_experiment_plan`` request; nothing is frozen or rendered.
(c) **stored-grant replay → freeze + render** — recording an
    ``ApprovalDecision(granted=True)`` for that same request id (obtained from
    the pending list of the first suspended run) replays store-first straight
    through the gate into ``freeze_experiment_plan`` (content-addressed: the same
    board freezes to the SAME id across a repeat run) and the
    ``plan_report_renderer`` render, surfaced as ``ModeResult.final_artifact``
    (kind ``plan_report``).

Run standalone with ``python regressions/plan-emergent-05c-orchestrator.py``;
the process exits 0 and the final line on success is
``plan-emergent-05c-orchestrator: ok``. ``orchestrator.run`` is async, so the
script drives it under ``asyncio.run(main())``.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from molexp.agent.loops import LoopHooks
from molexp.harness import (
    ApprovalPendingError,
    FileArtifactStore,
    HarnessRunContext,
    ModeResult,
    SQLiteApprovalStore,
)
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.modes.emergent_plan import EmergentPlanOrchestrator, PlanLoopRunner
from molexp.harness.plan import (
    FROZEN_PLAN_KIND,
    BoardTask,
    Difficulty,
    FeasibilityAnnotation,
    TaskBoard,
    board_path,
    write_board,
)
from molexp.harness.schemas import ApprovalDecision
from molexp.harness.stages import auto_grant_approver
from molexp.workspace import Run, Workspace

SLUG = "plan-emergent-05c-orchestrator"

# A rich draft carrying an experiment title + objective so the spec half is
# always form-valid and only the task board drives valid/invalid.
_USER_INPUT = (
    '{"title": "Zwitterion CG", '
    '"objective": "Coarse-grain a zwitterionic lipid and measure area-per-lipid."}'
)


# ── boards ───────────────────────────────────────────────────────────────────


def _valid_board() -> TaskBoard:
    """A form-VALID board: one task with acceptance + a feasibility annotation."""
    return TaskBoard(
        version=1,
        tasks=(
            BoardTask(
                id="t1",
                name="build coarse-grained system",
                acceptance=("system energy is finite",),
                feasibility=FeasibilityAnnotation(
                    reachable=True, difficulty=Difficulty.TRIVIAL, rationale="stub"
                ),
            ),
        ),
    )


def _malformed_board() -> TaskBoard:
    """A form-INVALID board: a task with NO acceptance criteria."""
    return TaskBoard(version=1, tasks=(BoardTask(id="t1", name="build", acceptance=()),))


# ── self-contained stub PlanLoopRunner ───────────────────────────────────────


class _CannedBoardRunner:
    """Stub :class:`PlanLoopRunner`: writes one canned board via ``write_board``.

    Stands in for the production ``InteractiveLoopPlanRunner`` (which drives a
    real Pi loop). It writes its canned board to ``board_path(ctx.workspace_root)``
    so the orchestrator reads it back — no real LLM, no loop, no subprocess.
    """

    def __init__(self, board: TaskBoard) -> None:
        self._board = board

    async def run_planning(
        self,
        *,
        ctx: HarnessRunContext,
        board: TaskBoard,
        tools: tuple[object, ...],
        hooks: LoopHooks,
        user_input: str,
    ) -> None:
        del board, tools, hooks, user_input
        write_board(board_path(ctx.workspace_root), self._board)


# ── fixtures / helpers ───────────────────────────────────────────────────────


def _make_run(root: Path, slug: str) -> Run:
    """Build a workspace ``Run`` exactly the way the orchestrator test does."""
    ws = Workspace(root=root / slug, name="lab")
    return (
        ws.add_project("p").add_experiment("e").add_run(params={"mode": "plan"}, id="emergent05c")
    )


def _gateway(run: Run) -> StubAgentGateway:
    """A stub gateway on the run's content-addressed store, with the renderer."""
    gw = StubAgentGateway(FileArtifactStore(root=run.run_dir / "artifacts"))
    gw.register(
        "plan_report_renderer",
        output={"title": "Plan report", "summary_md": "# Plan\n\nlooks good"},
        output_kind="plan_report",
        raw_text="rendered plan report",
    )
    return gw


def _store(run: Run) -> FileArtifactStore:
    return FileArtifactStore(root=run.run_dir / "artifacts")


def _approvals(run: Run) -> SQLiteApprovalStore:
    return SQLiteApprovalStore(run.run_dir / "harness.sqlite")


# ── observations ─────────────────────────────────────────────────────────────


async def _check_guard_no_human_surface(root: Path) -> None:
    """(a) A malformed final board is rejected before any human gate is opened."""
    # The self-contained stub is a structural ``PlanLoopRunner`` — the seam the
    # orchestrator depends on (production injects ``InteractiveLoopPlanRunner``).
    assert isinstance(_CannedBoardRunner(_valid_board()), PlanLoopRunner), (
        "the stub runner must satisfy the runtime-checkable PlanLoopRunner Protocol"
    )

    run = _make_run(root, "guard")
    orch = EmergentPlanOrchestrator(
        loop_runner=_CannedBoardRunner(_malformed_board()), approve=auto_grant_approver
    )
    # A persistently malformed board must be rejected (raise) — but must never
    # open a human gate: no review pack, no freeze, no pending approval.
    with contextlib.suppress(Exception):
        await orch.run(run=run, user_input=_USER_INPUT, gateway=_gateway(run))

    store = _store(run)
    review_pack = store.latest_by_kind("review_pack")
    frozen = store.latest_by_kind(FROZEN_PLAN_KIND)
    pending = _approvals(run).pending(run.id)
    print(f"[obs-a] malformed board: review_pack={review_pack} frozen={frozen} pending={pending}")
    assert review_pack is None, "a malformed board must not persist a review_pack"
    assert frozen is None, "a malformed board must not be frozen"
    assert pending == [], "a malformed board must open no human approval request"


async def _check_store_first_suspend(root: Path) -> None:
    """(b) No approver + no grant ⇒ ApprovalPendingError + a pending request."""
    run = _make_run(root, "suspend")
    orch = EmergentPlanOrchestrator(loop_runner=_CannedBoardRunner(_valid_board()), approve=None)
    suspended = False
    try:
        await orch.run(run=run, user_input=_USER_INPUT, gateway=_gateway(run))
    except ApprovalPendingError:
        suspended = True
    assert suspended, "a valid board with no approver must suspend store-first"

    pending = _approvals(run).pending(run.id)
    intents = [req.intent for req in pending]
    print(
        f"[obs-b] suspended={suspended} pending_intents={intents} "
        f"request_ids={[req.id for req in pending]}"
    )
    assert pending, "a pending approval request must be recorded on suspend"
    assert pending[0].intent == "approve_experiment_plan", (
        "the pending request must be the approve_experiment_plan gate"
    )
    store = _store(run)
    assert store.latest_by_kind(FROZEN_PLAN_KIND) is None, "nothing is frozen on suspend"
    assert store.latest_by_kind("plan_report") is None, "nothing is rendered on suspend"


async def _check_stored_grant_replay(root: Path) -> None:
    """(c) A recorded grant replays store-first → freeze (stable id) + render."""
    run = _make_run(root, "replay")
    gw = _gateway(run)
    orch = EmergentPlanOrchestrator(loop_runner=_CannedBoardRunner(_valid_board()), approve=None)
    # First run suspends store-first; obtain the gate's request id publicly.
    with contextlib.suppress(ApprovalPendingError):
        await orch.run(run=run, user_input=_USER_INPUT, gateway=gw)

    approvals = _approvals(run)
    [pending] = approvals.pending(run.id)
    approvals.record_decision(
        ApprovalDecision(
            request_id=pending.id,
            granted=True,
            decided_by="ui-operator",
            decided_at=datetime(2026, 7, 22, tzinfo=UTC),
            reason="looks correct",
        )
    )

    # Second run replays the stored grant → freeze + render.
    result = await orch.run(run=run, user_input=_USER_INPUT, gateway=gw)
    store = _store(run)
    frozen = store.latest_by_kind(FROZEN_PLAN_KIND)
    assert isinstance(result, ModeResult), "orchestrator.run must return a ModeResult"
    assert frozen is not None, "a granted replay must freeze the plan"
    assert result.final_artifact is not None, "a replay must surface a final artifact"
    print(f"[obs-c] replay: frozen_id={frozen.id} final_artifact_kind={result.final_artifact.kind}")
    assert result.final_artifact.kind == "plan_report", (
        "the final artifact must be the plan_report_renderer render output"
    )

    # Third run on the SAME board yields the SAME content-addressed frozen id.
    await orch.run(run=run, user_input=_USER_INPUT, gateway=gw)
    frozen_again = store.latest_by_kind(FROZEN_PLAN_KIND)
    assert frozen_again is not None
    print(
        f"[obs-c] repeat replay: frozen_id={frozen_again.id} (stable={frozen_again.id == frozen.id})"
    )
    assert frozen_again.id == frozen.id, (
        "the same board must freeze to the same content-addressed id"
    )


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-plan-emergent-05c-"))
    try:
        await _check_guard_no_human_surface(root)
        await _check_store_first_suspend(root)
        await _check_stored_grant_replay(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"{SLUG}: ok")


if __name__ == "__main__":
    asyncio.run(main())
