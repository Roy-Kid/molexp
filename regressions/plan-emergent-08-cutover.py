"""Regression: the emergent-plan cutover is complete (plan-emergent-08-cutover).

Binding runtime example for spec ``plan-emergent-08-cutover``. It proves the
cutover onto :class:`~molexp.harness.modes.plan_orchestrator.PlanOrchestrator`
with two orthogonal checks, both offline, deterministic, stub-driven — no
network, no subprocess, no server, no CLI, no real LLM:

1. **Absence guard** — the retired mode/stage machinery is gone. Each deleted
   module (``molexp.harness.mode`` / ``molexp.harness.modes.plan`` /
   ``molexp.harness.stages.repair_loop`` /
   ``molexp.harness.stages.sequential_task_build``) raises
   ``ModuleNotFoundError`` on import, and the harness public surface no longer
   advertises ``"Mode"`` / ``"PlanMode"`` — it advertises
   ``"PlanOrchestrator"`` instead, at exactly 22 symbols.

2. **Shared-path drive** — a plan is driven END-TO-END through the ONE public
   shared entry point (:func:`~molexp.services.plan_runtime.drive_plan_mode`,
   the single way CLI and server run a plan pipeline) onto the new
   orchestrator, importing NO deleted symbol. A self-contained stub
   ``PlanLoopRunner`` writes a canned VALID board via ``write_board``; an
   ``auto_grant_approver`` grants the human gate in one pass so the drive
   completes without suspending. The result is a ``ModeResult`` whose phase
   artifacts are present — a content-addressed ``frozen_experiment_plan``
   artifact was written and ``ModeResult.final_artifact`` is the
   ``plan_report_renderer`` render output (kind ``plan_report``).

The construction mirrors ``regressions/plan-emergent-05c-orchestrator.py``
exactly (the same ``Workspace(...).add_project(...).add_experiment(...).add_run(...)``
Run, the same self-contained ``_CannedBoardRunner`` writing a canned board, the
same ``StubAgentGateway`` with a registered ``plan_report_renderer``).

Run standalone with ``python regressions/plan-emergent-08-cutover.py``; the
process exits 0 and the final line on success is ``plan-emergent-08-cutover: ok``.
``drive_plan_mode`` is async, so the script drives it under ``asyncio.run(main())``.
"""

from __future__ import annotations

import importlib
import shutil
import tempfile
from pathlib import Path

import molexp.harness as harness
from molexp.agent.loops import LoopHooks
from molexp.harness import (
    FileArtifactStore,
    HarnessRunContext,
    ModeResult,
    SQLiteApprovalStore,  # noqa: F401 — imported to prove the public surface still carries it
)
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.modes.plan_orchestrator import PlanOrchestrator
from molexp.harness.plan import (
    FROZEN_PLAN_KIND,
    BoardTask,
    Difficulty,
    FeasibilityAnnotation,
    TaskBoard,
    board_path,
    write_board,
)
from molexp.harness.stages import auto_grant_approver
from molexp.services.plan_runtime import drive_plan_mode
from molexp.workspace import Run, Workspace

SLUG = "plan-emergent-08-cutover"

# Modules the cutover deleted; importing any of them must fail hard.
_DELETED_MODULES = (
    "molexp.harness.mode",
    "molexp.harness.modes.plan",
    "molexp.harness.stages.repair_loop",
    "molexp.harness.stages.sequential_task_build",
)

# A rich draft carrying an experiment title + objective so the spec half is
# always form-valid and only the task board drives valid/invalid.
_USER_INPUT = (
    '{"title": "Zwitterion CG", '
    '"objective": "Coarse-grain a zwitterionic lipid and measure area-per-lipid."}'
)


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


# ── boards / fixtures ────────────────────────────────────────────────────────


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


def _make_run(root: Path, slug: str) -> Run:
    """Build a workspace ``Run`` exactly the way the orchestrator test does."""
    ws = Workspace(root=root / slug, name="lab")
    return ws.add_project("p").add_experiment("e").add_run(params={"mode": "plan"}, id="emergent08")


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


# ── check 1: absence guard ───────────────────────────────────────────────────


def _check_absence_guard() -> None:
    """The retired mode/stage machinery is deleted and de-listed."""
    for name in _DELETED_MODULES:
        raised = False
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            raised = True
        assert raised, f"deleted module {name!r} must raise ModuleNotFoundError on import"

    all_symbols = harness.__all__
    assert "Mode" not in all_symbols, "retired 'Mode' must not be in molexp.harness.__all__"
    assert "PlanMode" not in all_symbols, "retired 'PlanMode' must not be in molexp.harness.__all__"
    assert "PlanOrchestrator" in all_symbols, (
        "the cutover target 'PlanOrchestrator' must be in molexp.harness.__all__"
    )
    assert len(all_symbols) == 22, (
        f"harness public surface must be 22 symbols, got {len(all_symbols)}"
    )

    print(
        f"[obs-1] absence guard: deleted={list(_DELETED_MODULES)} "
        f"__all__={len(all_symbols)} symbols (Mode/PlanMode absent, "
        f"PlanOrchestrator present)"
    )


# ── check 2: shared-path end-to-end drive ────────────────────────────────────


async def _check_shared_path_drive(root: Path) -> None:
    """A plan is driven end-to-end through drive_plan_mode onto the orchestrator."""
    run = _make_run(root, "cutover")
    gateway = _gateway(run)
    orchestrator = PlanOrchestrator(
        loop_runner=_CannedBoardRunner(_valid_board()),
        approve=auto_grant_approver,
        # Phase 1 only: this script's contract is "offline, no subprocess", and
        # phase-2 realization runs codegen agents through an executor subprocess.
        realize=False,
    )

    result = await drive_plan_mode(
        orchestrator,
        run=run,
        user_input=_USER_INPUT,
        gateway=gateway,
        capability_registry=None,
    )

    assert isinstance(result, ModeResult), "drive_plan_mode must return a ModeResult"
    assert result.run_id == run.id, "the ModeResult must carry the driven run's id"
    assert run.status == "succeeded", (
        "the shared drive_plan_mode path must mark the plan run 'succeeded'"
    )

    store = FileArtifactStore(root=run.run_dir / "artifacts")
    frozen = store.latest_by_kind(FROZEN_PLAN_KIND)
    assert frozen is not None, "a completed drive must persist a frozen_experiment_plan artifact"
    assert result.final_artifact is not None, "a completed drive must surface a final artifact"
    assert result.final_artifact.kind == "plan_report", (
        "ModeResult.final_artifact must be the plan_report_renderer render output"
    )

    print(
        f"drove PlanOrchestrator via drive_plan_mode; "
        f"frozen={frozen.id} final={result.final_artifact.kind}"
    )


async def main() -> None:
    _check_absence_guard()

    root = Path(tempfile.mkdtemp(prefix="molexp-plan-emergent-08-"))
    try:
        await _check_shared_path_drive(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"{SLUG}: ok")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
