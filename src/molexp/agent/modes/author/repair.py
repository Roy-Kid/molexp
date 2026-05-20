"""``PlanDiff``-centric repair for AuthorMode's per-task debug loop.

Every debug iteration that ends in a failure produces a
:class:`~molexp.agent.modes._planning.PlanDiff` — never a blind re-run.
A repair confined to one task's *source* (the common case: the impl was
wrong, the test was right) carries a ``replace`` op on that step and is
applied locally; a repair that needs a plan *shape* change escalates the
``PlanDiff`` against the whole :class:`~molexp.agent.modes._planning.PlanGraph`
via :meth:`PlanDiff.apply`.

This module owns:

- :func:`build_repair_diff` — turn a pytest failure into a typed
  ``PlanDiff`` with a populated ``failed_invariant`` / ``affected_nodes``.
- :func:`apply_local_repair` — record a single-task repair (the source
  is rewritten in place; the diff is the audit record).
- :func:`escalate_plan_repair` — apply a plan-shape ``PlanDiff`` to the
  ``PlanGraph`` and return the new graph.

Pure data + pure functions; no LLM, no I/O.
"""

from __future__ import annotations

from molexp.agent.modes._planning import (
    DiffOpKind,
    PlanDiff,
    PlanGraph,
    PlanNodeOp,
    PlanStep,
)

__all__ = [
    "apply_local_repair",
    "build_repair_diff",
    "escalate_plan_repair",
]

_TEST_FAILURE_INVARIANT = "generated_task_test_passes"
"""The invariant a per-task pytest failure violates — the generated
implementation must make its generated test pass."""


def build_repair_diff(
    *,
    plan_graph: PlanGraph,
    step_id: str,
    traceback: str,
    attempt: int,
) -> PlanDiff:
    """Build a :class:`PlanDiff` describing one debug-loop repair.

    The diff names the failed invariant
    (``generated_task_test_passes``), the affected step, and — when the
    step exists in the plan — a ``replace`` op carrying the same step
    (the source is rewritten out-of-band; the op records that the step's
    materialized code is being replaced). The downstream dependents of
    ``step_id`` are listed as ``invalidated``.

    Args:
        plan_graph: The plan being materialized.
        step_id: The id of the failing task/step.
        traceback: The pytest traceback (folded into the rationale).
        attempt: 1-based debug-loop attempt number.

    Returns:
        A frozen :class:`PlanDiff`.
    """
    step = plan_graph.step_by_id(step_id)
    operations: tuple[PlanNodeOp, ...] = ()
    if step is not None:
        operations = (PlanNodeOp(kind=DiffOpKind.replace, node_id=step_id, step=step),)
    rationale = (
        f"attempt {attempt}: task {step_id!r} test failed; "
        f"repair its implementation. Traceback tail: {_traceback_tail(traceback)}"
    )
    return PlanDiff(
        failed_invariant=_TEST_FAILURE_INVARIANT,
        affected_nodes=(step_id,),
        operations=operations,
        rationale=rationale,
        reused=tuple(s.id for s in plan_graph.steps if s.id != step_id),
        invalidated=plan_graph.downstream_of(step_id),
    )


def apply_local_repair(diff: PlanDiff, *, plan_graph: PlanGraph) -> PlanGraph:
    """Apply a single-task repair locally — the plan shape is unchanged.

    A local repair rewrites one task's source on disk (out-of-band) and
    keeps the ``PlanGraph`` topology intact. The ``diff`` is the audit
    record; applying it returns a graph with the same steps (a
    ``replace`` of a step by itself is a no-op on topology). The input
    graph is never mutated.

    Raises:
        ValueError: when the diff is not a single-task local repair
            (it adds/removes a step) — use :func:`escalate_plan_repair`.
    """
    if not is_local_repair(diff):
        raise ValueError(
            "apply_local_repair received a plan-shape diff; call escalate_plan_repair instead"
        )
    return diff.apply(plan_graph)


def escalate_plan_repair(diff: PlanDiff, *, plan_graph: PlanGraph) -> PlanGraph:
    """Apply a plan-shape repair to the ``PlanGraph`` via :meth:`PlanDiff.apply`.

    Used when a repair cannot be confined to one task's source — a step
    must be added, removed, or structurally replaced. Returns the new
    graph; the input is never mutated.
    """
    return diff.apply(plan_graph)


def is_local_repair(diff: PlanDiff) -> bool:
    """Return whether ``diff`` is confined to one task's source.

    A local repair touches exactly one node and either has no ops or a
    single ``replace`` op on that node by an identical-id step.
    """
    if len(diff.affected_nodes) != 1:
        return False
    node_id = diff.affected_nodes[0]
    for op in diff.operations:
        if op.kind is not DiffOpKind.replace or op.node_id != node_id:
            return False
        if op.step is not None and op.step.id != node_id:
            return False
    return True


def _ensure_step(step: PlanStep) -> PlanStep:
    """Identity helper kept for symmetry with future repair shapes."""
    return step


def _traceback_tail(traceback: str, *, max_chars: int = 400) -> str:
    """Return the last ``max_chars`` of a traceback, single-lined."""
    flattened = " ".join(traceback.split())
    if len(flattened) <= max_chars:
        return flattened
    return "…" + flattened[-max_chars:]
