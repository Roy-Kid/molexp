"""Tests for :func:`run_plan` (plan-emergent-05c + phase-2 wiring).

Offline, stub-driven: a ``StubAgentGateway`` plus a local stub
``draft=`` seam that writes a canned board. Phase 2 is disabled
(``realize=False``) so these unit tests do not require codegen agents.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from molexp.harness import (
    ApprovalPendingError,
    FileArtifactStore,
    ModeResult,
    SQLiteApprovalStore,
)
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.modes.plan import run_plan
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

_USER_INPUT = (
    '{"title": "Zwitterion CG", '
    '"objective": "Coarse-grain a zwitterionic lipid and measure area-per-lipid."}'
)


def _valid_board() -> TaskBoard:
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
    return TaskBoard(
        version=1,
        tasks=(BoardTask(id="t1", name="build", acceptance=()),),
    )


class _CannedDraft:
    def __init__(self, board: TaskBoard) -> None:
        self._board = board
        self.calls = 0

    async def __call__(self, *, ctx: Any, user_input: str) -> None:
        del user_input
        self.calls += 1
        write_board(board_path(ctx.workspace_root), self._board)


@pytest.fixture()
def run(tmp_path: Path):
    ws = Workspace(root=tmp_path / "ws", name="lab")
    return ws.add_project("p").add_experiment("e").add_run(params={"mode": "plan"}, id="plan05c")


def _gateway(run: Any) -> StubAgentGateway:
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


class TestStoreBundle:
    async def test_run_reproduces_the_build_ctx_store_bundle(self, run: Any) -> None:
        result = await run_plan(
            run=run,
            user_input=_USER_INPUT,
            gateway=_gateway(run),
            draft=_CannedDraft(_valid_board()),
            approve=auto_grant_approver,
            realize=False,
        )

        assert isinstance(result, ModeResult)
        assert result.run_id == run.id
        artifacts_dir = run.run_dir / "artifacts"
        assert artifacts_dir.is_dir() and any(artifacts_dir.iterdir())
        assert (run.run_dir / "harness.sqlite").is_file()
        store = _store(run)
        assert store.latest_by_kind("review_pack") is not None
        assert store.latest_by_kind(FROZEN_PLAN_KIND) is not None
        assert store.latest_by_kind("plan_report") is not None


class TestGuardFailSteersBack:
    async def test_malformed_final_board_never_reaches_the_gate(self, run: Any) -> None:
        with contextlib.suppress(Exception):
            await run_plan(
                run=run,
                user_input=_USER_INPUT,
                gateway=_gateway(run),
                draft=_CannedDraft(_malformed_board()),
                approve=auto_grant_approver,
                realize=False,
            )

        store = _store(run)
        assert store.latest_by_kind("review_pack") is None
        assert store.latest_by_kind(FROZEN_PLAN_KIND) is None
        assert _approvals(run).pending(run.id) == []

    async def test_form_loop_retries_until_board_is_valid(self, run: Any) -> None:
        """Malformed first draft is a workflow continue, not a review subject."""
        boards = [_malformed_board(), _valid_board()]

        async def draft(*, ctx: Any, user_input: str) -> None:
            del user_input
            write_board(board_path(ctx.workspace_root), boards.pop(0) if boards else _valid_board())

        result = await run_plan(
            run=run,
            user_input=_USER_INPUT,
            gateway=_gateway(run),
            draft=draft,
            approve=auto_grant_approver,
            realize=False,
            board_max_iters=4,
        )
        assert isinstance(result, ModeResult)
        assert _store(run).latest_by_kind(FROZEN_PLAN_KIND) is not None


class TestStoreFirstSuspend:
    async def test_valid_board_suspends_store_first_without_approver(self, run: Any) -> None:
        with pytest.raises(ApprovalPendingError):
            await run_plan(
                run=run,
                user_input=_USER_INPUT,
                gateway=_gateway(run),
                draft=_CannedDraft(_valid_board()),
                approve=None,
                realize=False,
            )

        pending = _approvals(run).pending(run.id)
        assert pending, "a pending approval request must be recorded on suspend"
        assert pending[0].intent == "approve_experiment_plan"
        store = _store(run)
        assert store.latest_by_kind(FROZEN_PLAN_KIND) is None
        # Plan book is rendered before the gate so the agent answer is filled
        # while still waiting for approval (freeze still post-grant only).
        assert store.latest_by_kind("plan_report") is not None


class TestStoredGrantReplay:
    async def test_stored_grant_replays_into_freeze_and_render(self, run: Any) -> None:
        gw = _gateway(run)
        kwargs = {
            "run": run,
            "user_input": _USER_INPUT,
            "gateway": gw,
            "draft": _CannedDraft(_valid_board()),
            "approve": None,
            "realize": False,
        }
        with pytest.raises(ApprovalPendingError):
            await run_plan(**kwargs)

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

        result = await run_plan(**kwargs)
        assert isinstance(result, ModeResult)
        store = _store(run)
        frozen = store.latest_by_kind(FROZEN_PLAN_KIND)
        assert frozen is not None
        assert result.final_artifact is not None
        assert result.final_artifact.kind == "plan_report"

        await run_plan(**kwargs)
        assert store.latest_by_kind(FROZEN_PLAN_KIND).id == frozen.id


class TestModeLikeShape:
    async def test_drive_plan_mode_returns_a_mode_result(self, run: Any) -> None:
        result = await drive_plan_mode(
            run=run,
            user_input=_USER_INPUT,
            gateway=_gateway(run),
            draft=_CannedDraft(_valid_board()),
            approve=auto_grant_approver,
            realize=False,
        )
        assert isinstance(result, ModeResult)
        assert run.status == "succeeded"


class TestPriorKnowledgeWire:
    """close-loop-01: AssembleKnowledgeContext on the live run_plan path."""

    async def test_assembles_knowledge_context_and_lineages_experiment_plan(self, run: Any) -> None:
        result = await run_plan(
            run=run,
            user_input=_USER_INPUT,
            gateway=_gateway(run),
            draft=_CannedDraft(_valid_board()),
            approve=auto_grant_approver,
            realize=False,
        )

        store = _store(run)
        knowledge = store.latest_by_kind("knowledge_context")
        plan = store.latest_by_kind("experiment_plan")
        assert knowledge is not None
        assert plan is not None
        assert knowledge.id in plan.parent_ids
        assert knowledge.id in {a.id for a in result.stage_artifacts}
        digest = store.get(knowledge.id).decode("utf-8")
        # Empty workspace still gets a uniform honest digest (no crash).
        assert digest.strip()

    async def test_failure_analysis_path_appears_in_digest(self, tmp_path: Path) -> None:
        from molexp.workspace.knowledge_item import KnowledgeItem, KnowledgeMeta, SourceRef

        ws = Workspace(root=tmp_path / "ws-fa", name="lab")
        ws.materialize()
        exp = ws.add_project("p").add_experiment("e")
        run = exp.add_run(params={"mode": "plan"}, id="plan-fa")
        item = KnowledgeItem(name="failure-grid")
        exp.add_folder(item)
        item.write_knowledge_meta(
            KnowledgeMeta(
                kind="FailureAnalysis",
                sources=[SourceRef(kind="run", ref=run.id)],
                created_by="test",
            )
        )
        item.write_index("grid too coarse near r_min — unique-fa-marker")

        await run_plan(
            run=run,
            user_input=_USER_INPUT,
            gateway=_gateway(run),
            draft=_CannedDraft(_valid_board()),
            approve=auto_grant_approver,
            realize=False,
        )

        store = FileArtifactStore(root=run.run_dir / "artifacts")
        knowledge = store.latest_by_kind("knowledge_context")
        assert knowledge is not None
        digest = store.get(knowledge.id).decode("utf-8")
        assert "unique-fa-marker" in digest
        assert "FailureAnalysis" in digest


class TestPlanLoopSystemPrompt:
    def test_appends_digest_when_present(self) -> None:
        from molexp.harness.modes.plan import plan_loop_system_prompt

        bare = plan_loop_system_prompt(None)
        with_digest = plan_loop_system_prompt("# Prior\n\npath: failure-x")
        assert bare in with_digest
        assert "failure-x" in with_digest
        assert "Prior knowledge" in with_digest
