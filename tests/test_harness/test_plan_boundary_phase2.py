"""Boundary 1 — Phase 2 realization is wired (unit, no live LLM / no e2e).

Pins:
* ``realize=False`` never calls realization and ends on ``plan_report``.
* ``realize=True`` calls ``_run_realization`` after freeze/report.
* ``materialize_plan_for_realization`` writes ``experiment_spec`` + ``bound_workflow``.
* Default constructor has ``realize=True`` (production posture).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from molexp.harness import FileArtifactStore, ModeResult
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.modes.plan import run_plan
from molexp.harness.plan import (
    FROZEN_PLAN_KIND,
    BoardTask,
    Difficulty,
    ExperimentPlan,
    FeasibilityAnnotation,
    TaskBoard,
    board_path,
    materialize_plan_for_realization,
    write_board,
)
from molexp.harness.plan.bind_board import board_plan_to_bound_workflow
from molexp.harness.schemas import PlanArtifactRef
from molexp.harness.stages import auto_grant_approver
from molexp.harness.stages.realize_board import RealizeBoard
from molexp.workspace import Workspace

_USER_INPUT = '{"title": "Boundary phase2", "objective": "Unit-test realization wiring."}'


def _valid_board() -> TaskBoard:
    return TaskBoard(
        version=1,
        tasks=(
            BoardTask(
                id="t1",
                name="build system",
                acceptance=("energy finite",),
                feasibility=FeasibilityAnnotation(
                    reachable=True, difficulty=Difficulty.TRIVIAL, rationale="stub"
                ),
            ),
        ),
    )


class _CannedDraft:
    def __init__(self, board: TaskBoard) -> None:
        self._board = board

    async def __call__(self, *, ctx: Any, user_input: str) -> None:
        del user_input
        write_board(board_path(ctx.workspace_root), self._board)


@pytest.fixture()
def run(tmp_path: Path):
    ws = Workspace(root=tmp_path / "ws", name="lab")
    return ws.add_project("p").add_experiment("e").add_run(params={"mode": "plan"}, id="bphase2")


def _gateway(run: Any) -> StubAgentGateway:
    gw = StubAgentGateway(FileArtifactStore(root=run.run_dir / "artifacts"))
    gw.register(
        "plan_report_renderer",
        output={"title": "Plan report", "summary_md": "# ok"},
        output_kind="plan_report",
    )
    return gw


class TestRealizeDefault:
    def test_run_plan_defaults_realize_true(self) -> None:
        import inspect

        assert inspect.signature(run_plan).parameters["realize"].default is True


class TestRealizeFalseSkipsPhase2:
    @pytest.mark.asyncio
    async def test_no_bound_workflow_and_final_is_plan_report(self, run: Any) -> None:
        result = await run_plan(
            run=run,
            user_input=_USER_INPUT,
            gateway=_gateway(run),
            draft=_CannedDraft(_valid_board()),
            approve=auto_grant_approver,
            realize=False,
        )
        store = FileArtifactStore(root=run.run_dir / "artifacts")
        assert store.latest_by_kind(FROZEN_PLAN_KIND) is not None
        assert store.latest_by_kind("plan_report") is not None
        assert store.latest_by_kind("bound_workflow") is None
        assert result.final_artifact is not None
        assert result.final_artifact.kind == "plan_report"


class TestRealizeTrueInvokesPhase2:
    @pytest.mark.asyncio
    async def test_run_realization_is_called_after_freeze(
        self, run: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        async def _fake_realize(self, ctx):
            calls.append("realize")
            return ctx.artifact_store.put_json(
                kind="execution_result",
                obj={"status": "succeeded", "metadata": {"mode": "compile", "stub": True}},
                created_by="test",
                parent_ids=[],
            )

        monkeypatch.setattr(RealizeBoard, "run", _fake_realize)
        result = await run_plan(
            run=run,
            user_input=_USER_INPUT,
            gateway=_gateway(run),
            draft=_CannedDraft(_valid_board()),
            approve=auto_grant_approver,
            realize=True,
        )
        assert calls == ["realize"]
        assert isinstance(result, ModeResult)
        assert result.final_artifact is not None
        assert result.final_artifact.kind == "execution_result"
        store = FileArtifactStore(root=run.run_dir / "artifacts")
        assert store.latest_by_kind(FROZEN_PLAN_KIND) is not None
        assert store.latest_by_kind("plan_report") is not None


class TestMaterializeBoundFromBoard:
    def test_board_plan_to_bound_workflow_maps_tasks(self) -> None:
        plan = ExperimentPlan(
            spec={"title": "t", "objective": "o"},
            board=_valid_board(),
        )
        bound = board_plan_to_bound_workflow(plan)
        assert len(bound.tasks) == 1
        assert bound.tasks[0].id == "t1"
        assert bound.tasks[0].ir_task_id == "t1"

    def test_materialize_writes_spec_and_bound(self, tmp_path: Path) -> None:
        store = FileArtifactStore(root=tmp_path / "artifacts")
        plan = ExperimentPlan(
            spec={"title": "t", "objective": "o"},
            board=_valid_board(),
        )
        spec_ref, bound_ref = materialize_plan_for_realization(plan, store, created_by="test")
        assert isinstance(spec_ref, PlanArtifactRef)
        assert isinstance(bound_ref, PlanArtifactRef)
        assert store.latest_by_kind("experiment_spec") is not None
        assert store.latest_by_kind("bound_workflow") is not None
