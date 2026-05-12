"""``drive_with_repair`` — outer Python loop for the PlanMode review→repair cycle.

The PlanMode pipeline (:data:`PLAN_WORKFLOW`) is a single-direction
13-node DAG; introducing reverse edges in the IR is rejected by
``Workflow.to_dict`` (see ``spec.py:480``). The repair loop therefore
sits *outside* the workflow as a plain Python ``while`` driver:

1. Run the current spec (the full ``PLAN_WORKFLOW`` on iteration 0; a
   :meth:`Workflow.subgraph` thereafter).
2. Inspect the terminal :class:`HandoffResult` to read the gate's
   :class:`~molexp.agent.review.ReviewDecision`.
3. Approved → return immediately.
4. Rejected + budget remains → archive the live artifact tree under
   ``<plan_id>/repairs/iter-<n>/``, persist the decision and update the
   manifest's ``repair_history``, build a partial-rerun subgraph from
   the rejection's ``target_steps`` / ``target_task_ids``, gather the
   boundary upstream values from the previous run as ``seed_outputs``,
   and loop.
5. Rejected + budget exhausted → emit
   :class:`~molexp.workflow.RepairBudgetExceeded`, force the final
   handoff's status to ``"rejected"``, and return.

Per-step :class:`StepRejected` exceptions raised by the
:class:`PlanTask` template method follow the same path as the
capability exceptions: synthesized into a :class:`ReviewDecision`,
archived, then routed through the partial-rerun machinery.

This module is the **only** consumer of
:meth:`Workflow.subgraph` + :meth:`Workflow.execute(seed_outputs=...)`
inside the agent layer; the per-task LLM filter is delivered through
:attr:`PlanDeps.repair_target_tasks` (read by
``GenerateTaskTests._execute`` / ``GenerateTaskImplementations._execute``).
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from mollog import get_logger

from molexp.agent.modes.plan._pipeline import PLAN_WORKFLOW
from molexp.agent.modes.plan.context import PlanRepairContext
from molexp.agent.modes.plan.errors import (
    CapabilityDiscoveryRequired,
    StepRejected,
    UnevidencedApiReference,
)
from molexp.agent.modes.plan.schemas import (
    HandoffResult,
    RepairIterationRecord,
    ReviewDecision,
)
from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.workflow import Workflow, WorkflowResult

if TYPE_CHECKING:
    from molexp.agent.modes.plan.protocols import PlanDeps

__all__ = ["RepairBudgetExceeded", "drive_with_repair"]


_LOG = get_logger(__name__)


class RepairBudgetExceeded(UserWarning):
    """Emitted by the PlanMode review→repair loop when the configured
    ``max_iterations`` budget is exhausted without the gate ever
    approving the materialized plan.

    Mirrors :class:`~molexp.workflow.LoopMaxItersExceeded` semantics: the
    workflow itself does not raise — the outer driver forces the final
    :class:`HandoffResult.status` to ``"rejected"`` and surfaces this
    warning so callers can detect "we ran out of repair budget" without
    having to inspect the manifest. Catch with
    ``pytest.warns(RepairBudgetExceeded)`` or filter via :mod:`warnings`.
    """


# Default plan-level nodes to re-run when the reviewer specifies only
# ``target_task_ids`` and no ``target_steps``: regenerate both codegen
# nodes (with the per-task filter) and cascade downstream so the
# validation + review pipeline reruns over the fresh artifacts.
_DEFAULT_TASK_LEVEL_NODES = ("GenerateTaskTests", "GenerateTaskImplementations")


async def drive_with_repair(
    deps: PlanDeps,
    user_input: str,
    *,
    max_iterations: int,
) -> WorkflowResult:
    """Run :data:`PLAN_WORKFLOW` with a structured review→repair loop.

    Args:
        deps: Initial :class:`PlanDeps`. Iteration state
            (``repair_target_tasks`` / ``repair_iteration``) is
            replaced per round; the rest is preserved unchanged.
        user_input: Top-level prompt fed to ``IngestReport``.
        max_iterations: Hard cap on completed review→repair cycles. The
            first run counts as iteration 0; the cap fires when the
            (iteration + 1)-th rejection arrives.

    Returns:
        The final :class:`WorkflowResult`. On approval, the result is
        the workflow's terminal output verbatim. On budget exhaustion,
        the ``HumanReview`` / ``FinalHandoffCheck`` payload's
        ``status`` is forced to ``"rejected"`` so downstream consumers
        observe the cap deterministically.
    """
    handle: PlanFolder = deps.plan_folder
    spec: Workflow = PLAN_WORKFLOW
    seed_outputs: dict[str, Any] | None = None
    iteration = 0
    unevidenced_count = 0
    repair_context = PlanRepairContext()

    _LOG.info(f"[plan] repair budget: {max_iterations} iteration(s)")
    while True:
        current_deps = replace(
            deps,
            repair_iteration=iteration,
            repair_context=repair_context,
        )
        seed_keys = list(seed_outputs.keys()) if seed_outputs else None
        if iteration:
            _LOG.info(f"[plan] repair iteration {iteration} start")
        _LOG.debug(f"[plan-repair] iter={iteration} running spec={spec.name} seed_keys={seed_keys}")
        try:
            result = await spec.execute(
                config={"user_input": user_input},
                deps=current_deps,
                seed_outputs=seed_outputs,
            )
        except (CapabilityDiscoveryRequired, UnevidencedApiReference, StepRejected) as exc:
            if isinstance(exc, StepRejected):
                decision = _synthesize_step_decision(exc)
            else:
                decision = _synthesize_capability_decision(exc, unevidenced_count=unevidenced_count)
                if isinstance(exc, UnevidencedApiReference):
                    unevidenced_count += 1
            _LOG.warning(
                f"[plan-repair] iter={iteration} {type(exc).__name__} → "
                f"target_steps={list(decision.target_steps)}"
            )
            if iteration + 1 >= max_iterations:
                _LOG.warning(
                    f"[plan-repair] iter={iteration} budget_exhausted on "
                    f"{type(exc).__name__}; surfacing the exception."
                )
                # Budget exhausted before any successful run → bubble
                # the exception so callers see a concrete failure
                # rather than a synthetic rejected handoff with no
                # `result.outputs`.
                handle.archive_artifacts_for_repair(iteration)
                handle.write_latest_decision(decision)
                _append_repair_history(handle, iteration, decision)
                raise

            # Archive the live tree, persist decision, advance iteration.
            handle.archive_artifacts_for_repair(iteration)
            handle.write_latest_decision(decision)
            _append_repair_history(handle, iteration, decision)

            if isinstance(exc, StepRejected):
                # Step-level rejection — PlanTask.execute stashed every
                # approved upstream output in ``deps.step_outputs_log``,
                # so we can replay only the rejected step (plus the
                # decision's cascade) instead of rewinding to the top.
                # The rejected step itself is intentionally NOT in the
                # log (its output was discarded when StepRejected was
                # raised), so the subgraph will recompute it.
                spec, seed_outputs = _next_round_inputs_from_log(deps.step_outputs_log, decision)
                repair_context = PlanRepairContext.from_decision(
                    iteration=iteration + 1,
                    decision=decision,
                    source=f"step:{exc.view.step_id}",
                )
                deps = replace(
                    deps,
                    repair_target_tasks=decision.target_task_ids or None,
                    repair_context=repair_context,
                )
                # Clear cached outputs for nodes the subgraph will
                # rebuild — otherwise stale entries shadow the rerun.
                _purge_log_for_subgraph(deps.step_outputs_log, spec)
                iteration += 1
                continue

            # Capability exceptions still have no partial outputs
            # attached, so rewind to the full pipeline. Earlier nodes
            # overwrite their already-written artefacts in place.
            spec = PLAN_WORKFLOW
            seed_outputs = None
            repair_context = PlanRepairContext.from_decision(
                iteration=iteration + 1,
                decision=decision,
                source="capability_gate",
            )
            deps = replace(
                deps,
                repair_target_tasks=None,
                repair_context=repair_context,
            )
            deps.step_outputs_log.clear()
            iteration += 1
            continue

        handoff = _terminal_handoff(result)
        if handoff is None:
            _LOG.error(f"[plan-repair] iter={iteration} aborted before reaching the review gate")
            # Pipeline failed before reaching the review gate; bail out.
            return result

        if handoff.decision.approved:
            _LOG.info(
                f"[plan] approved: status={handoff.status} ready_for_run={handoff.ready_for_run}"
            )
            return result

        # Rejected — check budget.
        if iteration + 1 >= max_iterations:
            _LOG.warning(
                f"[plan-repair] iter={iteration} budget_exhausted "
                f"max_iterations={max_iterations} — forcing rejected"
            )
            warnings.warn(
                RepairBudgetExceeded(
                    f"PlanMode repair loop reached max_iterations={max_iterations} "
                    f"without approval. Forcing HandoffResult.status='rejected'."
                ),
                stacklevel=2,
            )
            return _force_rejected(result, handoff)

        # Archive the live tree, persist the decision, update the manifest.
        decision = handoff.decision
        _LOG.info(
            "[plan] review requested repair: "
            f"steps={list(decision.target_steps) or 'default'} "
            f"tasks={list(decision.target_task_ids) or 'all'}"
        )
        handle.archive_artifacts_for_repair(iteration)
        handle.write_latest_decision(decision)
        _append_repair_history(handle, iteration, decision)

        # Build the next iteration's spec + seeds.
        spec, seed_outputs = _next_round_inputs(result, decision)
        repair_context = PlanRepairContext.from_decision(
            iteration=iteration + 1,
            decision=decision,
            source="final_review",
        )
        deps = replace(
            deps,
            repair_target_tasks=decision.target_task_ids or None,
            repair_context=repair_context,
        )
        iteration += 1


# ── Helpers ────────────────────────────────────────────────────────────────


def _synthesize_capability_decision(
    exc: CapabilityDiscoveryRequired | UnevidencedApiReference,
    *,
    unevidenced_count: int,
) -> ReviewDecision:
    """Build a synthetic :class:`ReviewDecision` from a capability-gate exception.

    Mapping (per spec § ``_repair_loop.drive_with_repair`` 失败映射):

    * :class:`CapabilityDiscoveryRequired` →
      ``target_steps=("DraftCapabilityNeeds", "DiscoverCapabilities")``,
      ``cascade_downstream=True``.
    * :class:`UnevidencedApiReference` →
      first occurrence ``("DiscoverCapabilities",)`` (unevidenced_count=0);
      second-and-later ``("DraftCapabilityNeeds", "DiscoverCapabilities")``.

    The decision is written to disk via
    :func:`PlanFolder.write_latest_decision` and embedded in
    the manifest's ``repair_history`` for telemetry, so the spec's
    target-step mapping survives in the audit log even when the
    pipeline is re-run from scratch.
    """
    if isinstance(exc, CapabilityDiscoveryRequired):
        return ReviewDecision(
            approved=False,
            reason=f"CapabilityDiscoveryRequired: {exc}",
            target_steps=("DraftCapabilityNeeds", "DiscoverCapabilities"),
            cascade_downstream=True,
            feedback=exc.detail or str(exc),
        )

    # UnevidencedApiReference
    if unevidenced_count == 0:
        target_steps = ("DiscoverCapabilities",)
    else:
        target_steps = ("DraftCapabilityNeeds", "DiscoverCapabilities")
    return ReviewDecision(
        approved=False,
        reason=f"UnevidencedApiReference: {exc}",
        target_steps=target_steps,
        cascade_downstream=True,
        feedback=(
            f"refs={list(exc.refs)} reason={exc.reason!r} "
            f"detail={exc.detail!r} unevidenced_count={unevidenced_count}"
        ),
    )


def _synthesize_step_decision(exc: StepRejected) -> ReviewDecision:
    """Promote a :class:`StepRejected` exception into a :class:`ReviewDecision`
    the repair-loop archive trail can persist.

    The exception already carries the per-step :class:`ReviewDecision`
    the policy returned; this helper ensures two invariants hold before
    the decision is fed to the subgraph selector:

    * ``target_steps`` is non-empty — defaults to the rejected step.
    * ``cascade_downstream=True`` — the rejected step's downstream
      artifacts are stale by definition, so they must be rerun.
      Honours an explicit ``cascade_downstream=True`` set by the policy
      but never silently drops the cascade for step rejections.
    """
    decision = exc.decision
    updates: dict[str, Any] = {}
    if not decision.target_steps:
        updates["target_steps"] = (exc.view.step_id,)
    if not decision.cascade_downstream:
        updates["cascade_downstream"] = True
    if not updates:
        return decision
    return decision.model_copy(update=updates)


def _terminal_handoff(result: WorkflowResult) -> HandoffResult | None:
    """Pick the most recent ``HandoffResult`` produced by the workflow.

    Prefers ``FinalHandoffCheck`` (the canonical terminal node); falls
    back to ``HumanReview`` when the pipeline did not advance past the
    review gate (e.g. validation failure that aborted the chain).
    """
    final = result.outputs.get("FinalHandoffCheck")
    if isinstance(final, HandoffResult):
        return final
    review = result.outputs.get("HumanReview")
    if isinstance(review, HandoffResult):
        return review
    return None


def _force_rejected(
    result: WorkflowResult,
    handoff: HandoffResult,
) -> WorkflowResult:
    """Return a :class:`WorkflowResult` whose terminal handoff is marked
    ``status='rejected'`` (used when ``max_iterations`` exhausts)."""
    rejected = handoff.model_copy(update={"status": "rejected", "ready_for_run": False})
    new_outputs = dict(result.outputs)
    if "FinalHandoffCheck" in new_outputs:
        new_outputs["FinalHandoffCheck"] = rejected
    if "HumanReview" in new_outputs:
        new_outputs["HumanReview"] = rejected
    return result.model_copy(update={"outputs": new_outputs})


def _append_repair_history(
    handle: PlanFolder,
    iteration: int,
    decision: ReviewDecision,
) -> None:
    """Bump ``manifest.repair_iterations`` and append a
    :class:`RepairIterationRecord` to ``manifest.repair_history``.

    Reads through :func:`_load_manifest_from_disk` so the existing
    ``handoff`` / ``plan_mode`` extension blocks (written by
    ``_persist_manifest_with_handoff`` in :mod:`tasks`) are stripped
    before the strict :class:`PlanManifest` validator runs.

    When the pipeline aborts before ``ValidateWorkspace`` writes a
    manifest (e.g. ``CapabilityDiscoveryRequired`` thrown from the
    middle of the DAG), this helper synthesizes a minimal stub via
    :func:`_build_manifest_stub` so the repair history still
    accumulates. Without that stub the iteration counter never
    increments and downstream telemetry loses the audit trail.
    """
    from molexp.agent.modes.plan.tasks import (
        _build_manifest_stub,
        _load_manifest_from_disk,
    )

    manifest_path = handle.manifest_path()
    manifest = _load_manifest_from_disk(manifest_path) if manifest_path.exists() else None
    if manifest is None:
        # Exception-driven recovery: no ValidateWorkspace ran yet, so
        # synthesize the minimal manifest the rest of the pipeline
        # expects. Subsequent ValidateWorkspace runs will preserve our
        # repair_iterations / repair_history fields via model_copy.
        manifest = _build_manifest_stub(handle.plan_id, handle.ir_dir() / "workflow.yaml")
    record = RepairIterationRecord(
        iteration=iteration,
        target_steps=decision.target_steps,
        target_task_ids=decision.target_task_ids,
        cascade_downstream=decision.cascade_downstream,
        archived_at=datetime.now(tz=UTC),
        feedback=decision.feedback,
    )
    new_manifest = manifest.model_copy(
        update={
            "repair_iterations": iteration + 1,
            "repair_history": (*manifest.repair_history, record),
        }
    )
    handle.write_manifest(new_manifest)


def _next_round_inputs_from_log(
    step_outputs_log: dict[str, Any],
    decision: ReviewDecision,
) -> tuple[Workflow, dict[str, Any]]:
    """Build the next iteration's subgraph + seed_outputs from a per-step log.

    Used when a :class:`StepRejected` exception interrupts the run
    before ``HumanReview`` produces a ``WorkflowResult`` — the rejected
    step's upstream outputs live in ``step_outputs_log`` (populated by
    :class:`PlanTask` after each approved step) rather than on a
    ``WorkflowResult.outputs`` mapping.

    Behaviour matches :func:`_next_round_inputs` otherwise: pick the
    target_steps from the decision, default ``cascade_downstream`` to
    ``True`` for step-level rejections (the rejected step's downstream
    almost always needs a rebuild), and require every boundary stub of
    the resulting subgraph to be present in the log.
    """
    target_steps = list(decision.target_steps)
    cascade = decision.cascade_downstream
    spec = PLAN_WORKFLOW.subgraph(target_steps, include_downstream=cascade)

    from molexp.workflow.spec import _BoundaryStubTask

    seed_outputs: dict[str, Any] = {}
    missing: list[str] = []
    for task_reg in spec._tasks:
        if not isinstance(task_reg.fn_or_class, _BoundaryStubTask):
            continue
        boundary_name = task_reg.name
        if boundary_name in step_outputs_log:
            seed_outputs[boundary_name] = step_outputs_log[boundary_name]
        else:
            missing.append(boundary_name)

    if missing:
        raise ValueError(
            f"drive_with_repair: cannot seed boundary upstream(s) {missing!r} from "
            f"step_outputs_log — these steps never approved on the previous "
            f"iteration. Available log keys: {sorted(step_outputs_log)}."
        )
    return spec, seed_outputs


def _purge_log_for_subgraph(step_outputs_log: dict[str, Any], spec: Workflow) -> None:
    """Drop log entries for steps the subgraph will recompute.

    The subgraph's *non-boundary* task names are the ones that will run
    fresh; their stale cached outputs would otherwise survive into the
    next iteration's seed if rejection happens again.
    """
    from molexp.workflow.spec import _BoundaryStubTask

    fresh_names = {
        task_reg.name
        for task_reg in spec._tasks
        if not isinstance(task_reg.fn_or_class, _BoundaryStubTask)
    }
    for name in fresh_names:
        step_outputs_log.pop(name, None)


def _next_round_inputs(
    prev_result: WorkflowResult,
    decision: ReviewDecision,
) -> tuple[Workflow, dict[str, Any]]:
    """Build the next iteration's subgraph spec + seed_outputs payload.

    When the rejection only specifies ``target_task_ids`` (per-task
    repair), the codegen pair plus its downstream cascade is selected so
    the per-task LLM filter actually runs. Boundary upstream values are
    pulled from ``prev_result.outputs`` to seed the partial-rerun.
    """
    target_steps = list(decision.target_steps)
    cascade = decision.cascade_downstream
    if not target_steps and decision.target_task_ids:
        target_steps = list(_DEFAULT_TASK_LEVEL_NODES)
        # Per-task repair always cascades — downstream needs to revalidate
        # against the regenerated artefacts.
        cascade = True

    spec = PLAN_WORKFLOW.subgraph(target_steps, include_downstream=cascade)

    # Boundary upstream tasks are registered on the subgraph as
    # ``_BoundaryStubTask`` instances (see ``Workflow.subgraph``); their
    # values must come from the previous run's outputs.
    from molexp.workflow.spec import _BoundaryStubTask

    seed_outputs: dict[str, Any] = {}
    missing: list[str] = []
    for task_reg in spec._tasks:
        if not isinstance(task_reg.fn_or_class, _BoundaryStubTask):
            continue
        boundary_name = task_reg.name
        if boundary_name in prev_result.outputs:
            seed_outputs[boundary_name] = prev_result.outputs[boundary_name]
        else:
            missing.append(boundary_name)

    if missing:
        raise ValueError(
            f"drive_with_repair: cannot seed boundary upstream(s) {missing!r} — "
            "they were absent from the previous workflow result. The rejection "
            "decision targets nodes whose upstream did not run successfully."
        )
    return spec, seed_outputs
