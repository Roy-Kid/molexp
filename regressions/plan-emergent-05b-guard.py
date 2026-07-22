"""Regression: the three pre-gate emergent-plan guard components, end-to-end and offline.

Binding runtime example for spec ``plan-emergent-05b-guard``. A library-external
user drives the **public** pre-gate guard surface — the reachability probe, the
form validator, and the plan review-pack builder — entirely from documented
imports. No network, no subprocess, no server, no CLI, no LLM; deterministic on
every run (the probe / validator / builder are pure sync functions and the
artifact store is content-addressed).

Three reference observations, each printed before it is asserted:

1. **reachability probe** — ``PlanReachabilityProbe().annotate(board, registry)``
   grades a board with one task whose name matches a registered capability and
   one that matches nothing: the matched task is ``reachable=True``, the
   unmatched task is ``reachable=False`` with ``Difficulty.HARD``. The input
   board is never mutated (a NEW board is returned).
2. **form validator** — a WELL-FORMED :class:`ExperimentPlan` (spec with
   title+objective; every task acceptance-complete and reachably probed) passes
   with zero ``error`` violations; wiping one task's acceptance flips
   ``passed`` to ``False`` and surfaces a ``task_missing_acceptance`` violation.
3. **plan review pack** — persisting the well-formed plan as a PRE-APPROVAL
   ``experiment_plan`` artifact and calling ``build_experiment_plan_review_pack``
   yields a ``policy == "hard"`` pack whose ``decision_options`` are
   ``["approve", "reject", "revise"]`` and whose ``subject_refs`` include the
   ``experiment_plan`` ref (NOT the frozen artifact — freeze is a post-approval
   consequence; ``freeze_experiment_plan`` is exercised separately only to show
   the frozen kind exists).

Run standalone with ``python regressions/plan-emergent-05b-guard.py``; the
process exits 0 and the final line on success is ``plan-emergent-05b-guard: ok``.
These APIs are synchronous — no asyncio involved.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from molexp.harness import (
    FileArtifactStore,
    HarnessRunContext,
    SQLiteArtifactLineageStore,
    SQLiteEventLog,
)
from molexp.harness.plan import (
    BoardTask,
    Difficulty,
    ExperimentPlan,
    TaskBoard,
    freeze_experiment_plan,
)
from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry
from molexp.harness.schemas import PlanArtifactRef, ReviewPack, ToolCapability
from molexp.harness.schemas.validation import PlanValidationReport
from molexp.harness.stages.plan_reachability_probe import PlanReachabilityProbe
from molexp.harness.stages.review_pack_builders import build_experiment_plan_review_pack
from molexp.harness.validators import EmergentPlanFormValidator

SLUG = "plan-emergent-05b-guard"

_SPEC: dict[str, str] = {
    "id": "exp-zwitterion",
    "title": "Zwitterion CG",
    "objective": "Coarse-grain a zwitterion in explicit water",
}


def _cap(cid: str, *, desc: str, side_effects: list[str] | None = None) -> ToolCapability:
    """Build a minimal :class:`ToolCapability` matched by substring search."""
    return ToolCapability(
        id=cid,
        package=cid.split(".")[0],
        name=cid.rsplit(".", 1)[-1],
        description=desc,
        input_schema={"type": "object"},
        output_schema={},
        side_effects=side_effects or [],
    )


def _task(task_id: str, name: str) -> BoardTask:
    """Build an acceptance-complete, as-yet-unprobed board task."""
    return BoardTask(id=task_id, name=name, acceptance=(f"{name} accepted",))


def _require_task(board: TaskBoard, task_id: str) -> BoardTask:
    """Look up a task that must exist, narrowing away the ``| None``."""
    task = board.task(task_id)
    assert task is not None, f"expected task {task_id!r} on the board"
    return task


def _error_count(report: PlanValidationReport) -> int:
    """Count only the blocking (``severity == "error"``) violations."""
    return sum(1 for v in report.violations if v.severity == "error")


def _check_reachability_probe() -> None:
    """Observation 1: a matched task is reachable; an unmatched task is HARD."""
    registry = InMemoryCapabilityRegistry([_cap("molpy.build.write", desc="build a polymer chain")])
    board = TaskBoard(tasks=(_task("t-build", "build"), _task("t-teleport", "quantum_teleport")))

    annotated = PlanReachabilityProbe().annotate(board, registry)

    reachable = _require_task(annotated, "t-build")
    unreachable = _require_task(annotated, "t-teleport")
    assert reachable.feasibility is not None, "reachable task lost its feasibility verdict"
    assert unreachable.feasibility is not None, "unreachable task lost its feasibility verdict"
    print(
        f"[obs-1] t-build reachable={reachable.feasibility.reachable} "
        f"difficulty={reachable.feasibility.difficulty}; "
        f"t-teleport reachable={unreachable.feasibility.reachable} "
        f"difficulty={unreachable.feasibility.difficulty}"
    )
    assert reachable.feasibility.reachable is True, "matched task must be reachable"
    assert unreachable.feasibility.reachable is False, "unmatched task must be unreachable"
    assert unreachable.feasibility.difficulty == Difficulty.HARD, (
        "an unmatched task must be graded HARD"
    )

    # Purity: the probe returns a NEW board and never mutates its input.
    assert annotated is not board, "annotate must return a fresh board object"
    assert board.tasks[0].feasibility is None, "annotate mutated the input board's task"


def _probed_plan(registry: InMemoryCapabilityRegistry) -> ExperimentPlan:
    """Build a well-formed plan: both tasks acceptance-complete and reachable."""
    board = TaskBoard(tasks=(_task("t-build", "build"), _task("t-solvate", "solvate")))
    probed = PlanReachabilityProbe().annotate(board, registry)
    return ExperimentPlan(spec=_SPEC, board=probed)


def _wipe_acceptance(plan: ExperimentPlan, task_id: str) -> ExperimentPlan:
    """Return a copy of ``plan`` with ``task_id``'s acceptance criteria removed."""
    tasks = tuple(
        task.model_copy(update={"acceptance": ()}) if task.id == task_id else task
        for task in plan.board.tasks
    )
    defect_board = plan.board.model_copy(update={"tasks": tasks})
    return plan.model_copy(update={"board": defect_board})


def _check_form_validator(plan: ExperimentPlan) -> None:
    """Observation 2: well-formed plan passes; a missing acceptance fails."""
    ok_report = EmergentPlanFormValidator.validate(plan)
    print(f"[obs-2] well-formed plan passed={ok_report.passed} errors={_error_count(ok_report)}")
    assert ok_report.passed is True, "a well-formed plan must pass the form validator"
    assert _error_count(ok_report) == 0, "a well-formed plan must have zero error violations"

    defect_report = EmergentPlanFormValidator.validate(_wipe_acceptance(plan, "t-build"))
    codes = sorted({v.code for v in defect_report.violations})
    print(f"[obs-2] acceptance-wiped plan passed={defect_report.passed} codes={codes}")
    assert defect_report.passed is False, "wiping an acceptance must fail the form validator"
    assert any(v.code == "task_missing_acceptance" for v in defect_report.violations), (
        "the defect must surface a task_missing_acceptance violation"
    )


def _make_ctx(root: Path) -> HarnessRunContext:
    """Assemble a harness run context around a fresh file/SQLite backing."""
    db = root / "harness.sqlite"
    artifacts = FileArtifactStore(root=root / "artifacts")
    return HarnessRunContext(
        run_id="run-plan-emergent-05b",
        workspace_root=root,
        artifact_store=artifacts,
        event_log=SQLiteEventLog(path=db),
        lineage_store=SQLiteArtifactLineageStore(path=db, artifact_store=artifacts),
    )


def _check_plan_review_pack(root: Path, plan: ExperimentPlan) -> None:
    """Observation 3: the pack is a hard review anchored on the experiment_plan."""
    ctx = _make_ctx(root)
    ref: PlanArtifactRef = ctx.artifact_store.put_json(
        kind="experiment_plan",
        obj=plan.model_dump(mode="json"),
        created_by="regression",
        parent_ids=[],
    )

    pack: ReviewPack = build_experiment_plan_review_pack(ctx)
    subject_ids = [r.id for r in pack.subject_refs]
    print(
        f"[obs-3] pack policy={pack.policy} options={pack.decision_options} "
        f"experiment_plan_ref={ref.id} subject_refs={subject_ids}"
    )
    assert pack.policy == "hard", "the plan review must be a hard gate"
    assert pack.decision_options == ["approve", "reject", "revise"], (
        "the plan review must offer approve/reject/revise"
    )
    assert any(r.id == ref.id for r in pack.subject_refs), (
        "the pack must anchor on the pre-approval experiment_plan artifact"
    )

    # Freeze exists as a separate, post-approval kind — but the pack does NOT
    # read it; the subject stays the pre-approval experiment_plan above.
    frozen: PlanArtifactRef = freeze_experiment_plan(
        plan, ctx.artifact_store, created_by="regression"
    )
    print(f"[obs-3] frozen kind={frozen.kind} id={frozen.id} (not a pack subject)")
    assert frozen.kind == "frozen_experiment_plan", "freeze must write the frozen kind"
    assert frozen.id not in subject_ids, "the frozen artifact must not be the pack subject"


def main() -> None:
    _check_reachability_probe()

    registry = InMemoryCapabilityRegistry(
        [
            _cap("molpy.build.write", desc="build a polymer chain"),
            _cap("molpy.solvate.run", desc="solvate the system in water"),
        ]
    )
    plan = _probed_plan(registry)
    _check_form_validator(plan)

    root = Path(tempfile.mkdtemp(prefix="molexp-plan-emergent-05b-"))
    try:
        _check_plan_review_pack(root, plan)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"{SLUG}: ok")


if __name__ == "__main__":
    main()
