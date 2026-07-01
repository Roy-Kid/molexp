"""``ProposalExecutor`` — the coordination-layer "Act" seam (integration.md §6.1, §8.3).

The seam a *granted* :class:`~molexp.harness.schemas.change_proposal.ChangeProposal`
dispatches through to actually perform its high-risk mutation ("the agent
proposes, the harness disposes"). This slice ships the reusable spine — the
``HighRiskOp`` → handler registry, the dispatch control-flow, the
``affected_objects`` binding guard, and the traceable-event recording — with a
handler Protocol but no concrete handler (curation handlers land in slice 02;
the gate wiring in slice 03).

Dispatch contract:

- **Unhandled op → loud.** A proposal whose ``proposed_change.op`` has no
  registered handler raises :class:`UnhandledHighRiskOpError` *before any
  effect* and records no event — a routing/config defect, never a silent no-op
  (§10 invariant #10).
- **Handler success → executed.** A resolved handler runs between a
  ``tool_called`` and a ``tool_completed`` event; its ``ProposalOutcome`` is
  returned verbatim.
- **Handler failure → recorded, not raised.** Any exception inside the handler
  (including an :class:`OutOfAffectedScopeError` scope violation) is caught,
  recorded as ``tool_failed``, and returned as
  ``ProposalOutcome(status="failed", reason=…)`` — §8.3's "recorded, not
  aborted". ``status`` stays within ``{executed, failed}`` (no schema change).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from molexp.harness.actions.recorder import ProposalActionRecorder
from molexp.harness.errors import OutOfAffectedScopeError, UnhandledHighRiskOpError
from molexp.harness.schemas.change_proposal import ProposalOutcome

if TYPE_CHECKING:
    from collections.abc import Iterable

    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.schemas.change_proposal import ChangeProposal, HighRiskOp, ObjectRef

__all__ = [
    "ChangeActionHandler",
    "ChangeActionRegistry",
    "ProposalExecutor",
    "assert_within_affected_scope",
]


@runtime_checkable
class ChangeActionHandler(Protocol):
    """Performs one :data:`HighRiskOp`'s effect for a granted proposal.

    A handler resolves the proposal's ``affected_objects`` to live targets,
    performs the mutation (bounded to that scope — see
    :func:`assert_within_affected_scope`), and returns an ``executed`` outcome.
    Raising is allowed: the :class:`ProposalExecutor` turns any exception into a
    recorded ``failed`` outcome.
    """

    async def apply(self, ctx: HarnessRunContext, proposal: ChangeProposal) -> ProposalOutcome: ...


class ChangeActionRegistry:
    """A ``HighRiskOp`` → :class:`ChangeActionHandler` registry.

    Modeled on :class:`~molexp.harness.registry.in_memory.InMemoryCapabilityRegistry`:
    a dict keyed by the op discriminator plus an insertion-order list.
    Registering an op that is already bound raises loudly (a wiring defect).
    """

    def __init__(self) -> None:
        self._by_op: dict[HighRiskOp, ChangeActionHandler] = {}
        self._order: list[HighRiskOp] = []

    def register(self, op: HighRiskOp, handler: ChangeActionHandler) -> None:
        if op in self._by_op:
            raise ValueError(f"a handler for high-risk op {op!r} is already registered")
        self._by_op[op] = handler
        self._order.append(op)

    def has(self, op: HighRiskOp) -> bool:
        return op in self._by_op

    def get(self, op: HighRiskOp) -> ChangeActionHandler:
        try:
            return self._by_op[op]
        except KeyError as exc:
            raise UnhandledHighRiskOpError(
                f"no ChangeActionHandler registered for high-risk op {op!r}"
            ) from exc


def assert_within_affected_scope(proposal: ChangeProposal, targets: Iterable[ObjectRef]) -> None:
    """Assert every *target* is in ``proposal.affected_objects`` (§8.2 binding scope).

    A handler calls this before touching any object so it can only mutate the
    objects the proposal was approved for. Returns ``None`` when all targets are
    in scope; raises :class:`OutOfAffectedScopeError` on the first target that
    is not.

    Args:
        proposal: The granted proposal whose ``affected_objects`` is the bound.
        targets: The object refs the handler is about to touch.
    """
    allowed = set(proposal.affected_objects)
    for target in targets:
        if target not in allowed:
            raise OutOfAffectedScopeError(
                f"target {target.kind}:{target.id} is outside the proposal's "
                f"affected_objects (proposal {proposal.id})"
            )


class ProposalExecutor:
    """Dispatches a granted proposal to its registered handler and records the attempt."""

    def __init__(self, registry: ChangeActionRegistry) -> None:
        self._registry = registry

    async def dispatch(self, ctx: HarnessRunContext, proposal: ChangeProposal) -> ProposalOutcome:
        """Run the handler for ``proposal.proposed_change.op`` and return its outcome.

        Raises:
            UnhandledHighRiskOpError: The op has no registered handler (loud,
                before any event or effect).
        """
        op = proposal.proposed_change.op
        if not self._registry.has(op):
            raise UnhandledHighRiskOpError(
                f"no ChangeActionHandler registered for high-risk op {op!r}"
            )
        handler = self._registry.get(op)

        ProposalActionRecorder.record_started(ctx.event_log, ctx.run_id, proposal)
        try:
            outcome = await handler.apply(ctx, proposal)
        except Exception as exc:
            reason = str(exc)
            ProposalActionRecorder.record_failed(ctx.event_log, ctx.run_id, proposal, reason)
            return ProposalOutcome(
                status="failed",
                decided_by="proposal-executor",
                decided_at=datetime.now(tz=UTC),
                reason=reason,
            )
        ProposalActionRecorder.record_completed(ctx.event_log, ctx.run_id, proposal, outcome)
        return outcome
