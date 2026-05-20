"""Diff cluster — the plan-diff repair unit and the approval gates.

:class:`PlanDiff` is the unit of the plan-diff-centric repair loop every
mode shares: it names the failed invariant, the affected nodes, the
proposed node operations, and what the repair reuses / invalidates.
:class:`ApprovalGate` enumerates the three layered human approvals.

This module references ``PlanStep`` / ``PlanGraph`` from the sibling
``plan_graph`` module, which in turn imports :class:`ApprovalGate` from
here. To break that cycle, ``diff`` never imports ``plan_graph`` at
module scope: the field references are forward refs resolved by
:func:`_resolve_forward_refs`, which the package ``__init__`` calls once
both modules are loaded.

Pure frozen-pydantic data models; no LLM, no I/O.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, model_validator

if TYPE_CHECKING:
    from .plan_graph import PlanGraph, PlanStep


class ApprovalGate(StrEnum):
    """The three layered human-approval gates.

    ``approve_direction`` is owned by PlanMode, ``approve_materialization``
    by AuthorMode, and ``approve_execution`` by RunMode.
    """

    approve_direction = "approve_direction"
    approve_materialization = "approve_materialization"
    approve_execution = "approve_execution"


class DiffOpKind(StrEnum):
    """The kind of operation a :class:`PlanNodeOp` performs."""

    add = "add"
    remove = "remove"
    replace = "replace"


class PlanNodeOp(BaseModel):
    """One add / remove / replace operation on a plan node.

    Attributes:
        kind: The operation kind.
        node_id: ``id`` of the plan step the operation targets.
        step: The new / replacement step; ``None`` for a ``remove``,
            required for ``add`` and ``replace``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: DiffOpKind
    node_id: str
    step: PlanStep | None

    @model_validator(mode="after")
    def _check_step_presence(self) -> PlanNodeOp:
        """Enforce that ``step`` is present iff the op adds or replaces."""
        if self.kind is DiffOpKind.remove and self.step is not None:
            raise ValueError("a 'remove' op must not carry a step")
        if self.kind in (DiffOpKind.add, DiffOpKind.replace) and self.step is None:
            raise ValueError(f"a '{self.kind.value}' op must carry a step")
        return self


class PlanDiff(BaseModel):
    """A proposed repair to a plan, as a set of node operations.

    Attributes:
        failed_invariant: The invariant whose violation triggered the
            repair.
        affected_nodes: ``id``s of the plan steps the repair touches.
        operations: The node operations the repair applies.
        rationale: Why this diff fixes the failed invariant.
        reused: ``id``s of steps the repair reuses unchanged.
        invalidated: ``id``s of steps the repair invalidates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    failed_invariant: str
    affected_nodes: tuple[str, ...]
    operations: tuple[PlanNodeOp, ...]
    rationale: str
    reused: tuple[str, ...]
    invalidated: tuple[str, ...]

    def apply(self, graph: PlanGraph) -> PlanGraph:
        """Return a new ``PlanGraph`` with the operations applied.

        ``add`` appends a step, ``remove`` drops the step matching
        ``node_id``, and ``replace`` swaps it in place. The order of
        unaffected steps is preserved. The input graph is never mutated.
        """
        from .plan_graph import PlanGraph as _PlanGraph

        steps = list(graph.steps)
        for op in self.operations:
            if op.kind is DiffOpKind.remove:
                steps = [step for step in steps if step.id != op.node_id]
            elif op.kind is DiffOpKind.replace:
                # op.step is guaranteed non-None by the model validator.
                replacement = op.step
                assert replacement is not None
                steps = [replacement if step.id == op.node_id else step for step in steps]
            else:  # DiffOpKind.add
                assert op.step is not None
                steps.append(op.step)
        return _PlanGraph(
            plan_id=graph.plan_id,
            intent_ref=graph.intent_ref,
            steps=tuple(steps),
            state=graph.state,
            compiled_contract_ref=graph.compiled_contract_ref,
            notes=graph.notes,
        )


def _resolve_forward_refs() -> None:
    """Resolve ``PlanNodeOp`` / ``PlanDiff`` forward references.

    Called once by the package ``__init__`` after every cluster module is
    imported. Injects ``PlanStep`` / ``PlanGraph`` into this module's
    namespace so :meth:`pydantic.BaseModel.model_rebuild` can resolve the
    deferred annotations — breaking the ``plan_graph`` <-> ``diff`` cycle.
    """
    from . import plan_graph

    globals()["PlanStep"] = plan_graph.PlanStep
    globals()["PlanGraph"] = plan_graph.PlanGraph
    PlanNodeOp.model_rebuild()
    PlanDiff.model_rebuild()
