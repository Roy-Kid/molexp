"""``drive_with_repair`` — outer Python loop for the PlanMode review→repair cycle.

The PlanMode pipeline (:data:`PLAN_WORKFLOW`) is a single-direction
11-node DAG; introducing reverse edges in the IR is rejected by
``Workflow.to_dict`` (see ``spec.py:480``). The repair loop therefore
sits *outside* the workflow as a plain Python ``while`` driver:

1. Run the current spec (the full ``PLAN_WORKFLOW`` on iteration 0; a
   :meth:`Workflow.subgraph` thereafter).
2. Inspect the terminal :class:`HandoffResult` to read the gate's
   :class:`ApprovalDecision`.
3. Approved → return immediately.
4. Rejected + budget remains → archive the live artifact tree under
   ``<plan_id>/repairs/iter-<n>/``, persist the decision and update the
   manifest's ``repair_history``, build a partial-rerun subgraph from
   the rejection's ``target_node_ids`` / ``target_task_ids``, gather the
   boundary upstream values from the previous run as ``seed_outputs``,
   and loop.
5. Rejected + budget exhausted → emit
   :class:`~molexp.workflow.RepairBudgetExceeded`, force the final
   handoff's status to ``"rejected"``, and return.

This module is the **only** consumer of
:meth:`Workflow.subgraph` + :meth:`Workflow.execute(seed_outputs=...)`
inside the agent layer; the per-task LLM filter is delivered through
:attr:`PlanDeps.repair_target_tasks` (read by
``GenerateTaskTests.execute`` / ``GenerateTaskImplementations.execute``).
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from mollog import get_logger

from molexp.agent.modes.plan._pipeline import PLAN_WORKFLOW
from molexp.agent.modes.plan.schemas import (
    ApprovalDecision,
    HandoffResult,
    RepairIterationRecord,
)
from molexp.agent.modes.plan.workspace_layout import PlanWorkspaceHandle
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
# ``target_task_ids`` and no ``target_node_ids``: regenerate both codegen
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
    handle: PlanWorkspaceHandle = deps.workspace_handle
    spec: Workflow = PLAN_WORKFLOW
    seed_outputs: dict[str, Any] | None = None
    iteration = 0

    _LOG.info(f"[plan-repair] start max_iterations={max_iterations} workspace={handle.root()}")
    while True:
        current_deps = replace(deps, repair_iteration=iteration)
        seed_keys = list(seed_outputs.keys()) if seed_outputs else None
        _LOG.info(f"[plan-repair] iter={iteration} running spec={spec.name} seed_keys={seed_keys}")
        result = await spec.execute(
            config={"user_input": user_input},
            deps=current_deps,
            seed_outputs=seed_outputs,
        )

        handoff = _terminal_handoff(result)
        if handoff is None:
            _LOG.error(f"[plan-repair] iter={iteration} aborted before reaching the review gate")
            # Pipeline failed before reaching the review gate; bail out.
            return result

        if handoff.decision.approved:
            _LOG.info(
                f"[plan-repair] iter={iteration} approved status={handoff.status} "
                f"ready_for_run={handoff.ready_for_run}"
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
            f"[plan-repair] iter={iteration} rejected "
            f"target_nodes={list(decision.target_node_ids)} "
            f"target_tasks={list(decision.target_task_ids)} "
            f"cascade={decision.cascade_downstream} — archiving"
        )
        handle.archive_artifacts_for_repair(iteration)
        handle.write_latest_decision(decision)
        _append_repair_history(handle, iteration, decision)

        # Build the next iteration's spec + seeds.
        spec, seed_outputs = _next_round_inputs(result, decision)
        deps = replace(
            deps,
            repair_target_tasks=decision.target_task_ids or None,
        )
        iteration += 1


# ── Helpers ────────────────────────────────────────────────────────────────


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
    handle: PlanWorkspaceHandle,
    iteration: int,
    decision: ApprovalDecision,
) -> None:
    """Bump ``manifest.repair_iterations`` and append a
    :class:`RepairIterationRecord` to ``manifest.repair_history``.

    Reads through :func:`_load_manifest_from_disk` so the existing
    ``handoff`` / ``plan_mode`` extension blocks (written by
    ``_persist_manifest_with_handoff`` in :mod:`tasks`) are stripped
    before the strict :class:`PlanManifest` validator runs.
    """
    from molexp.agent.modes.plan.tasks import _load_manifest_from_disk

    manifest_path = handle.manifest_path()
    if not manifest_path.exists():
        # Pipeline aborted before ValidateWorkspace wrote a manifest. Do
        # not invent one — the archive on disk is still useful.
        return
    manifest = _load_manifest_from_disk(manifest_path)
    if manifest is None:
        return
    record = RepairIterationRecord(
        iteration=iteration,
        target_node_ids=decision.target_node_ids,
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


def _next_round_inputs(
    prev_result: WorkflowResult,
    decision: ApprovalDecision,
) -> tuple[Workflow, dict[str, Any]]:
    """Build the next iteration's subgraph spec + seed_outputs payload.

    When the rejection only specifies ``target_task_ids`` (per-task
    repair), the codegen pair plus its downstream cascade is selected so
    the per-task LLM filter actually runs. Boundary upstream values are
    pulled from ``prev_result.outputs`` to seed the partial-rerun.
    """
    target_nodes = list(decision.target_node_ids)
    cascade = decision.cascade_downstream
    if not target_nodes and decision.target_task_ids:
        target_nodes = list(_DEFAULT_TASK_LEVEL_NODES)
        # Per-task repair always cascades — downstream needs to revalidate
        # against the regenerated artefacts.
        cascade = True

    spec = PLAN_WORKFLOW.subgraph(target_nodes, include_downstream=cascade)

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
