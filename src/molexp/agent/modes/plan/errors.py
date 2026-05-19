"""Domain exceptions for the PlanMode pipeline.

- :class:`SkeletonCompileError` — ``GenerateWorkflowSkeleton`` raises
  when the LLM-emitted Python source fails :func:`compile`. Hard
  error; bubbles out of the workflow.
- :class:`CapabilityDiscoveryRequired` — raised inside capability
  probes when discovery is impossible. Caught by
  :class:`~molexp.agent.modes.plan.tasks.PlanTask` and converted to a
  :class:`~molexp.agent.modes.plan.state.RepairSignal`.
- :class:`UnevidencedApiReference` — raised inside
  :func:`~molexp.agent.modes.plan.tasks._gate_evidence` when a codegen
  module references an API absent from the evidence batch. Caught at
  the :class:`PlanTask` boundary.
- :class:`StepRejected` — retained for back-compat; pipeline tasks no
  longer raise it (per-step rejections plant a signal directly).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molexp.workflow import WorkflowError

if TYPE_CHECKING:
    from molexp.agent.review import ReviewDecision, StepView

__all__ = [
    "CapabilityDiscoveryRequired",
    "SkeletonCompileError",
    "StepRejected",
    "UnevidencedApiReference",
]


class SkeletonCompileError(RuntimeError):
    """Generated workflow skeleton failed :func:`compile` validation.

    Raised by ``GenerateWorkflowSkeleton`` when the LLM-emitted source
    cannot be parsed. The triggering :class:`SyntaxError` is chained
    through ``raise … from exc`` so callers can inspect
    ``error.__cause__`` to locate the offending line.

    Attributes:
        path: The would-be file path of the failed source. Useful for
            error messages even though no file was written.
    """

    def __init__(self, message: str, *, path: str = "") -> None:
        self.path = path
        super().__init__(message)


class CapabilityDiscoveryRequired(WorkflowError):
    """Capability discovery is needed but cannot proceed.

    Raised by ``NullCapabilityProbe.discover`` when its input report
    sets ``discovery_required=True`` (no source configured, source
    unavailable, or the probe declined the request). The
    ``planmode-review-repair-loop`` driver maps this exception to a
    re-run of ``DraftCapabilityNeeds`` + ``DiscoverCapabilities`` with
    ``cascade_downstream=True``.

    Attributes:
        reason: Short slug describing why discovery is blocked (e.g.
            ``"no probe configured"``). Logged by the repair loop.
        detail: Free-form human-readable expansion of ``reason``.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str = "",
        detail: str = "",
    ) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(message)


class UnevidencedApiReference(WorkflowError):
    """A generated module references an API not covered by evidence.

    Raised by the three codegen nodes (``GenerateWorkflowSkeleton`` /
    ``GenerateTaskTests`` / ``GenerateTaskImplementations``) when either
    the LLM's ``evidence_refs`` schema field disagrees with the
    source's ``__capability_evidence__`` literal
    (``reason="declared_block_mismatch"``) or
    :func:`~molexp.agent.modes.plan.capability.validate_codegen_evidence`
    returns a non-empty miss set (``reason=""``).

    The ``planmode-review-repair-loop`` driver maps this exception to
    a re-run of ``DiscoverCapabilities`` on the first occurrence and
    escalates to ``DraftCapabilityNeeds`` + ``DiscoverCapabilities`` on
    the second.

    Attributes:
        refs: The offending dotted-path identifiers (the same shape as
            :attr:`~molexp.agent.modes.plan.capability.CapabilityEvidence.api_ref`).
        reason: Short slug; ``"declared_block_mismatch"`` for the
            schema-vs-source check, empty when raised after the
            post-codegen AST diff.
        detail: Free-form human-readable expansion.
    """

    def __init__(
        self,
        message: str,
        *,
        refs: tuple[str, ...] = (),
        reason: str = "",
        detail: str = "",
    ) -> None:
        self.refs = refs
        self.reason = reason
        self.detail = detail
        super().__init__(message)


class StepRejected(WorkflowError):
    """A per-step :class:`~molexp.agent.review.ReviewPolicy` returned
    ``approved=False``.

    Raised by :class:`~molexp.agent.modes.plan.tasks.PlanTask` after a
    node's ``_execute()`` returns but before its result is propagated
    downstream.  The repair driver maps this exception to a re-run of
    ``decision.target_steps`` (defaulting to the rejected step itself
    when the tuple is empty), cascading downstream if
    ``decision.cascade_downstream`` is True.

    Attributes:
        view: The :class:`~molexp.agent.review.StepView` the policy was
            asked to review.  Surfaces ``step_id`` for the repair loop.
        decision: The :class:`~molexp.agent.review.ReviewDecision`
            returned by the policy — its ``feedback`` and target fields
            drive the next iteration.
    """

    def __init__(self, view: StepView, decision: ReviewDecision) -> None:
        self.view = view
        self.decision = decision
        message = f"step {view.step_id!r} rejected by review policy: reason={decision.reason!r}"
        super().__init__(message)
