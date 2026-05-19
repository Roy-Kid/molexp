"""PlanMode outer-loop tasks: bookends + 5 stage drivers.

The outer workflow's body chains seven tasks: :class:`PrepareIteration`
(loop body[0], resets per-iteration state) → five stage tasks
(:class:`UnderstandStage` / :class:`StrategizeStage` / :class:`BindStage` /
:class:`MaterializeStage` / :class:`VerifyStage`, each driving its
small sub-workflow from :mod:`molexp.agent.modes.plan._pipeline`) →
:class:`RepairDecide` (loop ``until`` task).

Stage tasks are plain :class:`Task` subclasses (not :class:`PlanTask`)
because their job is orchestration, not the domain work itself —
they don't write their own ``*_result.yaml`` and don't run the
per-step review policy. The wrapped leaves inside each sub-workflow
are :class:`PlanTask` subclasses and carry their own signal /
policy / checkpoint plumbing.

Cross-stage data flows two ways:

* Single payload — passed as ``ctx.inputs`` to the next stage.
* Multi-payload boundary seeding — the stage task pulls upstream
  values from :attr:`PlanRuntimeState.last_inner_outputs` and seeds
  the sub-workflow's :class:`_BoundaryStub` nodes via
  ``seed_outputs=``.

Iteration control belongs to ``wf.loop(max_iters=N)``; the stage
tasks never short-circuit the outer loop themselves.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from mollog import get_logger

from molexp.agent.modes.plan.context import PlanRepairContext, format_node_label
from molexp.agent.modes.plan.errors import (
    CapabilityDiscoveryRequired,
    UnevidencedApiReference,
)
from molexp.agent.modes.plan.plan_folder import RepairIterationRecord
from molexp.agent.modes.plan.protocols import PlanDeps
from molexp.agent.modes.plan.schemas import HandoffResult
from molexp.agent.modes.plan.state import RepairSignal
from molexp.agent.review import ReviewDecision
from molexp.workflow import Task, TaskContext, Workflow, WorkflowResult
from molexp.workflow.types import Next

__all__ = [
    "BindStage",
    "MaterializeStage",
    "PrepareIteration",
    "RepairDecide",
    "StrategizeStage",
    "UnderstandStage",
    "VerifyStage",
]


_LOG = get_logger(__name__)


# ── PrepareIteration ──────────────────────────────────────────────────────


class PrepareIteration(Task):
    """First task of every outer-loop body.

    Resets :attr:`PlanRuntimeState.repair_signal` so the new iteration
    starts clean; on iteration ≥ 1, archives the previous round's
    materialised artefacts under ``<plan_id>/repairs/iter-<n>/``.

    Wired to the first stage via ``wf.control`` so no payload flows
    out — :class:`UnderstandStage` starts with ``ctx.inputs=None``.
    """

    async def execute(self, ctx: TaskContext[Any, PlanDeps, None]) -> None:
        runtime = ctx.deps.runtime
        runtime.repair_signal = None

        if runtime.iteration > 0:
            try:
                ctx.deps.plan_folder.archive_artifacts_for_repair(runtime.iteration - 1)
            except Exception:  # noqa: BLE001 — archive failures are non-fatal
                _LOG.warning(
                    f"[plan] PrepareIteration: archive for iter {runtime.iteration - 1} failed",
                    exc_info=True,
                )
        _LOG.info(
            f"[plan] {format_node_label('PrepareIteration')} ready iteration={runtime.iteration}"
        )
        return None


# ── Stage-task helpers ────────────────────────────────────────────────────


async def _drive_subgraph(
    *,
    stage_name: str,
    subgraph: Workflow,
    deps: PlanDeps,
    config: Any,  # noqa: ANN401 — JSONMapping
    seed_outputs: dict[str, Any] | None = None,
) -> WorkflowResult | None:
    """Run a sub-workflow on behalf of a stage task.

    Returns the :class:`WorkflowResult` on success, ``None`` if a
    capability gate raised before any inner task could plant a signal
    (extremely rare — most gates plant via :class:`PlanTask.execute`).
    Either way, the caller observes the gate via
    ``deps.runtime.repair_signal`` and short-circuits accordingly.
    """
    _LOG.info(
        f"[plan] {format_node_label(stage_name)} start iteration={deps.runtime.iteration}"
    )
    try:
        return await subgraph.execute(
            config=config,
            deps=deps,
            seed_outputs=seed_outputs,
        )
    except (CapabilityDiscoveryRequired, UnevidencedApiReference) as exc:
        # Probe / gate raised before PlanTask.execute could catch it
        # (e.g. raised inside a non-PlanTask helper).
        from molexp.agent.modes.plan.tasks import _signal_from_capability_exc

        deps.runtime.repair_signal = _signal_from_capability_exc(
            exc, source_step=stage_name, runtime=deps.runtime
        )
        _LOG.warning(
            f"[plan] {format_node_label(stage_name)} caught {type(exc).__name__}; "
            "planted repair signal"
        )
        return None


def _stage_should_skip(stage_name: str, runtime: Any) -> bool:  # noqa: ANN401
    """Return ``True`` when an earlier stage planted a repair signal."""
    signal = runtime.repair_signal
    if signal is None:
        return False
    _LOG.debug(
        f"[plan] {format_node_label(stage_name)} skip — "
        f"repair signaled by {signal.source_step!r}"
    )
    return True


# ── UnderstandStage ───────────────────────────────────────────────────────


class UnderstandStage(Task):
    """Stage 1/5 — intake + summarize + clarify.

    Drives :data:`INTAKE_SUBGRAPH` (3 leaves, no boundaries).
    Returns :class:`ClarificationResult` for downstream stages.
    """

    async def execute(self, ctx: TaskContext[Any, PlanDeps, None]) -> Any:  # noqa: ANN401
        runtime = ctx.deps.runtime
        if _stage_should_skip("UnderstandStage", runtime):
            return None

        from molexp.agent.modes.plan._pipeline import INTAKE_SUBGRAPH

        result = await _drive_subgraph(
            stage_name="UnderstandStage",
            subgraph=INTAKE_SUBGRAPH,
            deps=ctx.deps,
            config=ctx.config,
        )
        if result is not None:
            runtime.last_inner_outputs.update(result.outputs)
        if runtime.repair_signal is not None:
            return None
        return result.outputs.get("ClarifyMissingInformation") if result else None


# ── StrategizeStage ───────────────────────────────────────────────────────


class StrategizeStage(Task):
    """Stage 2/5 — draft the natural-language implementation plan.

    A single leaf (:class:`DraftImplementationPlan`) — no sub-workflow
    overhead. The leaf reads :class:`ClarificationResult` from
    ``ctx.inputs`` (passed by UnderstandStage via the outer
    ``depends_on``) and returns :class:`PlanBriefResult`.
    """

    async def execute(self, ctx: TaskContext[Any, PlanDeps, Any]) -> Any:  # noqa: ANN401
        runtime = ctx.deps.runtime
        if _stage_should_skip("StrategizeStage", runtime):
            return None

        from molexp.agent.modes.plan.tasks import DraftImplementationPlan

        _LOG.info(
            f"[plan] {format_node_label('StrategizeStage')} start "
            f"iteration={runtime.iteration}"
        )
        leaf_ctx = TaskContext(
            state=ctx.state,
            deps=ctx.deps,
            inputs=ctx.inputs,
            config=ctx.config,
        )
        result = await DraftImplementationPlan().execute(leaf_ctx)
        if result is not None:
            runtime.last_inner_outputs["DraftImplementationPlan"] = result
        if runtime.repair_signal is not None:
            return None
        return result


# ── BindStage ─────────────────────────────────────────────────────────────


class BindStage(Task):
    """Stage 3/5 — capability discovery (draft needs + resolve evidence).

    Drives :data:`DISCOVERY_SUBGRAPH` (2 leaves + 1 boundary). Seeds
    the ``ClarifyMissingInformation`` boundary with the upstream brief —
    the discovery service uses both the report digest (via the plan
    folder on disk) and the freshly drafted plan brief.
    """

    async def execute(self, ctx: TaskContext[Any, PlanDeps, Any]) -> Any:  # noqa: ANN401
        runtime = ctx.deps.runtime
        if _stage_should_skip("BindStage", runtime):
            return None

        from molexp.agent.modes.plan._pipeline import DISCOVERY_SUBGRAPH

        plan_brief = ctx.inputs
        result = await _drive_subgraph(
            stage_name="BindStage",
            subgraph=DISCOVERY_SUBGRAPH,
            deps=ctx.deps,
            config=ctx.config,
            seed_outputs={"ClarifyMissingInformation": plan_brief},
        )
        if result is not None:
            runtime.last_inner_outputs.update(result.outputs)
        if runtime.repair_signal is not None:
            return None
        return result.outputs.get("DiscoverCapabilities") if result else None


# ── MaterializeStage ──────────────────────────────────────────────────────


class MaterializeStage(Task):
    """Stage 4/5 — IR compilation + codegen.

    Drives :data:`MATERIALIZE_SUBGRAPH` (5 leaves + 2 boundaries).
    Seeds the ``DraftImplementationPlan`` and ``DiscoverCapabilities``
    boundaries with the plan brief and evidence batch accumulated by
    earlier stages on :attr:`runtime.last_inner_outputs`.
    """

    async def execute(self, ctx: TaskContext[Any, PlanDeps, Any]) -> Any:  # noqa: ANN401
        runtime = ctx.deps.runtime
        if _stage_should_skip("MaterializeStage", runtime):
            return None

        from molexp.agent.modes.plan._pipeline import MATERIALIZE_SUBGRAPH

        plan_brief = runtime.last_inner_outputs.get("DraftImplementationPlan")
        evidence = ctx.inputs  # from BindStage
        result = await _drive_subgraph(
            stage_name="MaterializeStage",
            subgraph=MATERIALIZE_SUBGRAPH,
            deps=ctx.deps,
            config=ctx.config,
            seed_outputs={
                "DraftImplementationPlan": plan_brief,
                "DiscoverCapabilities": evidence,
            },
        )
        if result is not None:
            runtime.last_inner_outputs.update(result.outputs)
        if runtime.repair_signal is not None:
            return None
        # MaterializeStage emits a synthetic marker; VerifyStage reads
        # the individual leaves it needs from runtime.last_inner_outputs.
        return result.outputs if result else None


# ── VerifyStage ───────────────────────────────────────────────────────────


class VerifyStage(Task):
    """Stage 5/5 — validate the workspace, run human review, finalize handoff.

    Drives :data:`VERIFY_SUBGRAPH` (3 leaves + 5 boundaries). Boundaries
    cover every cross-stage upstream the verify leaves reference by
    name; seed values come from :attr:`runtime.last_inner_outputs`.
    Returns the :class:`HandoffResult` from FinalHandoffCheck.
    """

    async def execute(self, ctx: TaskContext[Any, PlanDeps, Any]) -> Any:  # noqa: ANN401
        runtime = ctx.deps.runtime
        if _stage_should_skip("VerifyStage", runtime):
            return None

        from molexp.agent.modes.plan._pipeline import VERIFY_SUBGRAPH

        outs = runtime.last_inner_outputs
        result = await _drive_subgraph(
            stage_name="VerifyStage",
            subgraph=VERIFY_SUBGRAPH,
            deps=ctx.deps,
            config=ctx.config,
            seed_outputs={
                "DraftReportDigest": outs.get("DraftReportDigest"),
                "DraftImplementationPlan": outs.get("DraftImplementationPlan"),
                "CompileTaskIR": outs.get("CompileTaskIR"),
                "GenerateTaskTests": outs.get("GenerateTaskTests"),
                "GenerateTaskImplementations": outs.get("GenerateTaskImplementations"),
            },
        )
        if result is not None:
            runtime.last_inner_outputs.update(result.outputs)
        if runtime.repair_signal is not None:
            return None
        handoff = result.outputs.get("FinalHandoffCheck") if result else None
        if not isinstance(handoff, HandoffResult):
            handoff = result.outputs.get("HumanReview") if result else None
        return handoff if isinstance(handoff, HandoffResult) else None


# ── RepairDecide ──────────────────────────────────────────────────────────


class RepairDecide(Task):
    """Loop ``until``-task. Reads gate outcome + handoff, returns ``Next(…)``.

    Three paths: a gate signal planted earlier in this iteration, an
    approved handoff (→ ``Next("exit")``), or a rejected handoff. The
    two non-approve paths share bookkeeping: promote to a
    :class:`ReviewDecision`, append a :class:`RepairIterationRecord`,
    persist through :class:`PlanFolder`, bump ``runtime.iteration``,
    return ``Next("continue")``.

    Budget enforcement is the workflow runtime's job — it forces
    ``Next("exit")`` and emits
    :class:`~molexp.workflow.LoopMaxItersExceeded` once ``max_iters``
    is reached.
    """

    async def execute(self, ctx: TaskContext[Any, PlanDeps, HandoffResult | None]) -> Any:  # noqa: ANN401
        runtime = ctx.deps.runtime
        handoff = ctx.inputs if isinstance(ctx.inputs, HandoffResult) else None

        signal = runtime.repair_signal
        if signal is None and handoff is not None and handoff.decision.approved:
            _LOG.info(
                f"[plan] RepairDecide: approved (iteration={runtime.iteration}); loop exiting"
            )
            return Next("exit")

        if signal is not None:
            decision = _decision_from_signal(signal)
        elif handoff is not None:
            decision = handoff.decision
        else:
            decision = ReviewDecision(
                approved=False,
                reason="pipeline failed before reaching FinalHandoffCheck",
                target_steps=(),
                target_task_ids=(),
                cascade_downstream=True,
                feedback="",
            )

        handle = ctx.deps.plan_folder
        handle.write_latest_decision(decision)

        record = RepairIterationRecord(
            iteration=runtime.iteration,
            target_steps=decision.target_steps,
            target_task_ids=decision.target_task_ids,
            cascade_downstream=decision.cascade_downstream,
            archived_at=datetime.now(tz=UTC),
            feedback=decision.feedback,
        )
        runtime.repair_history.append(record)
        runtime.latest_decision = decision
        runtime.repair_context = PlanRepairContext.from_decision(
            iteration=runtime.iteration + 1,
            decision=decision,
            source=signal.source_kind if signal is not None else "final_review",
        )

        try:
            manifest = handle.load_or_create_manifest()
            new_manifest = manifest.model_copy(
                update={
                    "repair_iterations": runtime.iteration + 1,
                    "repair_history": (*manifest.repair_history, record),
                }
            )
            # ``_write_raw_manifest`` preserves the ``handoff`` / ``plan_mode``
            # extension sections HumanReview / FinalHandoffCheck wrote.
            # ``write_manifest`` would silently drop them.
            handle._write_raw_manifest(new_manifest.model_dump(mode="json"))
        except Exception:  # noqa: BLE001 — telemetry; never block loop progress
            _LOG.warning(
                "[plan] RepairDecide: failed to persist manifest history; "
                "continuing the loop",
                exc_info=True,
            )

        runtime.iteration += 1
        _LOG.info(
            "[plan] RepairDecide: requesting repair "
            f"iter→{runtime.iteration} signal={signal.source_kind if signal else 'final_review'}"
        )
        return Next("continue")


def _decision_from_signal(signal: RepairSignal) -> ReviewDecision:
    """Promote a task-planted :class:`RepairSignal` into a :class:`ReviewDecision`."""
    return ReviewDecision(
        approved=False,
        reason=signal.reason or f"{signal.source_kind} from {signal.source_step}",
        target_steps=signal.target_steps,
        target_task_ids=signal.target_task_ids,
        cascade_downstream=signal.cascade_downstream,
        feedback=signal.feedback,
    )
