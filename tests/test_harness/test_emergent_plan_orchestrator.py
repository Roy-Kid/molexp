"""RED tests for the emergent-planning orchestrator (plan-emergent-05c).

``EmergentPlanOrchestrator`` (``molexp.harness.modes.emergent_plan``, not yet
written) composes the phases 01-05b parts into one runnable planning pipeline
**without** subclassing ``Mode`` or using a ``.mode_ledger``. These tests pin
its four binding behaviors, all offline and stub-driven — a ``StubAgentGateway``
plus a **local stub** ``PlanLoopRunner`` whose ``run_planning`` merely writes a
canned board via ``board_store`` (no real LLM, no real Pi loop, no subprocess):

* ac-001 — the orchestrator reproduces the ``Mode._build_ctx`` store bundle
  (``FileArtifactStore`` + ``SQLiteEventLog`` / ``SQLiteArtifactLineageStore`` /
  ``SQLiteApprovalStore`` on ``run_dir/harness.sqlite`` + gateway + registry).
* ac-002 — a **malformed** final board never reaches the human gate: the
  deterministic ``EmergentPlanFormValidator`` guard blocks before the
  ``StepAuditLoop`` so no ``review_pack`` and no pending ``ApprovalRequest`` are
  recorded. (The injected ``should_stop`` guard is also asserted to deny a
  malformed board at the loop level.)
* ac-003 — a **valid** board reaches the ``StepAuditLoop`` hard gate and, with
  no approver + no stored grant, suspends store-first (``ApprovalPendingError``
  + a pending ``approve_experiment_plan`` request); nothing is frozen/rendered.
* ac-004 — a stored grant replays store-first into ``freeze_experiment_plan``
  (content-addressed: same board ⇒ same frozen id) + the ``plan_report_renderer``
  render, surfaced as ``ModeResult.final_artifact``.
* ac-006 — the orchestrator satisfies the shared ``drive_plan_mode`` ``_ModeLike``
  shape (``async run(*, run, user_input, gateway, capability_registry=None)``).

All of this is RED until ``molexp.harness.modes.emergent_plan`` exists.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from molexp.agent.loops import LoopHooks, LoopState
from molexp.harness import (
    ApprovalPendingError,
    FileArtifactStore,
    ModeResult,
    SQLiteApprovalStore,
)
from molexp.harness.gateways.stub import StubAgentGateway

# RED: the module under test does not exist yet.
from molexp.harness.modes.emergent_plan import (
    EmergentPlanOrchestrator,
    PlanLoopRunner,
)
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
from molexp.services.plan_runtime import drive_plan_mode
from molexp.workspace import Workspace

pytestmark = pytest.mark.asyncio

# A rich draft carrying an experiment title + objective so whichever way the
# orchestrator sources the spec (JSON-parse or free-text), the spec is form-valid
# and only the task board drives valid/invalid.
_USER_INPUT = (
    '{"title": "Zwitterion CG", '
    '"objective": "Coarse-grain a zwitterionic lipid and measure area-per-lipid."}'
)


# ── boards ─────────────────────────────────────────────────────────────────


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
    """A form-INVALID board: a task with **no** acceptance criteria."""
    return TaskBoard(
        version=1,
        tasks=(BoardTask(id="t1", name="build", acceptance=()),),
    )


# ── local stub PlanLoopRunner ──────────────────────────────────────────────


class _CannedBoardRunner:
    """Stub ``PlanLoopRunner``: writes one canned board via ``board_store``.

    Stands in for the production ``InteractiveLoopPlanRunner`` (which drives a
    real Pi loop). It captures the injected ``hooks`` / ``tools`` so a test can
    assert the orchestrator wired a ``should_stop`` guard, and writes its canned
    board to ``board_path(ctx.workspace_root)`` so the orchestrator reads it back.
    """

    def __init__(self, board: TaskBoard) -> None:
        self._board = board
        self.calls = 0
        self.captured_hooks: LoopHooks | None = None
        self.captured_tools: tuple[Any, ...] = ()

    async def run_planning(
        self, *, ctx: Any, board: Any, tools: Any, hooks: LoopHooks, user_input: str
    ) -> None:
        del board, user_input
        self.calls += 1
        self.captured_hooks = hooks
        self.captured_tools = tuple(tools)
        write_board(board_path(ctx.workspace_root), self._board)


# ── fixtures / helpers ─────────────────────────────────────────────────────


@pytest.fixture()
def run(tmp_path: Path):
    ws = Workspace(root=tmp_path / "ws", name="lab")
    return (
        ws.add_project("p").add_experiment("e").add_run(params={"mode": "plan"}, id="emergent05c")
    )


def _gateway(run: Any) -> StubAgentGateway:
    """A stub gateway on the run's content-addressed store, with the renderer."""
    gw = StubAgentGateway(FileArtifactStore(root=run.run_dir / "artifacts"))
    gw.register(
        "plan_report_renderer",
        output={"title": "Plan report", "summary_md": "# Plan\n\nlooks good"},
        output_kind="plan_report",
        raw_text="rendered plan report",
    )
    return gw


def _store(run: Any) -> FileArtifactStore:
    return FileArtifactStore(root=run.run_dir / "artifacts")


def _approvals(run: Any) -> SQLiteApprovalStore:
    return SQLiteApprovalStore(run.run_dir / "harness.sqlite")


class TestPlanLoopRunnerProtocol:
    def test_stub_runner_satisfies_the_runtime_checkable_protocol(self) -> None:
        assert isinstance(_CannedBoardRunner(_valid_board()), PlanLoopRunner)


class TestStoreBundle:
    async def test_run_reproduces_the_build_ctx_store_bundle(self, run: Any) -> None:
        """ac-001: artifacts land under run_dir/artifacts; sqlite bundle on harness.sqlite."""
        orch = EmergentPlanOrchestrator(
            loop_runner=_CannedBoardRunner(_valid_board()), approve=auto_grant_approver
        )
        result = await orch.run(run=run, user_input=_USER_INPUT, gateway=_gateway(run))

        assert isinstance(result, ModeResult)
        assert result.run_id == run.id
        artifacts_dir = run.run_dir / "artifacts"
        assert artifacts_dir.is_dir() and any(artifacts_dir.iterdir())
        assert (run.run_dir / "harness.sqlite").is_file()
        store = _store(run)
        # The full compose ran through the store bundle: review pack, freeze, render.
        assert store.latest_by_kind("review_pack") is not None
        assert store.latest_by_kind(FROZEN_PLAN_KIND) is not None
        assert store.latest_by_kind("plan_report") is not None


class TestGuardFailSteersBack:
    async def test_malformed_final_board_never_reaches_the_gate(self, run: Any) -> None:
        """ac-002: the deterministic form guard blocks before the human gate."""
        orch = EmergentPlanOrchestrator(
            loop_runner=_CannedBoardRunner(_malformed_board()), approve=auto_grant_approver
        )
        # A persistently malformed board must be rejected (raise) or steered back,
        # but must never open a human gate.
        with contextlib.suppress(Exception):
            await orch.run(run=run, user_input=_USER_INPUT, gateway=_gateway(run))

        store = _store(run)
        assert store.latest_by_kind("review_pack") is None
        assert store.latest_by_kind(FROZEN_PLAN_KIND) is None
        assert _approvals(run).pending(run.id) == []

    async def test_injected_should_stop_guard_denies_a_malformed_board(self, run: Any) -> None:
        """ac-002: the orchestrator injects a should_stop guard that vetoes a malformed board."""
        stub = _CannedBoardRunner(_valid_board())
        orch = EmergentPlanOrchestrator(loop_runner=stub, approve=auto_grant_approver)
        await orch.run(run=run, user_input=_USER_INPUT, gateway=_gateway(run))

        assert stub.captured_hooks is not None
        guard = stub.captured_hooks.should_stop
        assert guard is not None
        # Overwrite the on-disk board with a malformed one; the guard reads the
        # current board (phase-04 read_board) and must deny termination.
        write_board(board_path(run.run_dir), _malformed_board())
        outcome = await guard(state=LoopState(step=1))
        assert outcome.is_deny


class TestStoreFirstSuspend:
    async def test_valid_board_suspends_store_first_without_approver(self, run: Any) -> None:
        """ac-003: no approver + no grant ⇒ ApprovalPendingError + a pending request."""
        orch = EmergentPlanOrchestrator(
            loop_runner=_CannedBoardRunner(_valid_board()), approve=None
        )
        with pytest.raises(ApprovalPendingError):
            await orch.run(run=run, user_input=_USER_INPUT, gateway=_gateway(run))

        pending = _approvals(run).pending(run.id)
        assert pending, "a pending approval request must be recorded on suspend"
        assert pending[0].intent == "approve_experiment_plan"
        store = _store(run)
        assert store.latest_by_kind(FROZEN_PLAN_KIND) is None
        assert store.latest_by_kind("plan_report") is None


class TestStoredGrantReplay:
    async def test_stored_grant_replays_into_freeze_and_render(self, run: Any) -> None:
        """ac-004: a recorded grant replays store-first → freeze (stable id) + render."""
        gw = _gateway(run)
        orch = EmergentPlanOrchestrator(
            loop_runner=_CannedBoardRunner(_valid_board()), approve=None
        )
        # First run suspends store-first.
        with pytest.raises(ApprovalPendingError):
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

        # Second run replays the grant → freeze + render.
        result = await orch.run(run=run, user_input=_USER_INPUT, gateway=gw)
        assert isinstance(result, ModeResult)
        store = _store(run)
        frozen = store.latest_by_kind(FROZEN_PLAN_KIND)
        assert frozen is not None
        assert result.final_artifact is not None
        assert result.final_artifact.kind == "plan_report"

        # Third run on the same board yields the SAME content-addressed frozen id.
        await orch.run(run=run, user_input=_USER_INPUT, gateway=gw)
        assert store.latest_by_kind(FROZEN_PLAN_KIND).id == frozen.id


class TestModeLikeShape:
    async def test_drive_plan_mode_returns_a_mode_result(self, run: Any) -> None:
        """ac-006: the orchestrator drops into the shared drive_plan_mode _ModeLike shape."""
        orch = EmergentPlanOrchestrator(
            loop_runner=_CannedBoardRunner(_valid_board()), approve=auto_grant_approver
        )
        result = await drive_plan_mode(orch, run=run, user_input=_USER_INPUT, gateway=_gateway(run))
        assert isinstance(result, ModeResult)
        assert run.status == "succeeded"
