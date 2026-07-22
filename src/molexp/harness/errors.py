"""Typed exceptions for ``molexp.harness``.

A single ``HarnessError`` base lets call sites catch every harness-raised
exception with one ``except`` clause; the leaf subclasses are what the
harness raises in practice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.harness.schemas import ApprovalRequest, PlanArtifactRef

__all__ = [
    "AgentResponseNotRegisteredError",
    "ApprovalPendingError",
    "ArtifactNotFoundError",
    "CapabilityAlreadyRegisteredError",
    "CapabilityCallValidationError",
    "CapabilityGapError",
    "CapabilityNotFoundError",
    "CapabilityResolutionError",
    "EventSeqConflictError",
    "HarnessError",
    "ObjectRefResolutionError",
    "OutOfAffectedScopeError",
    "StageExecutionError",
    "StagePersistedFailureError",
    "UnhandledHighRiskOpError",
]


class HarnessError(Exception):
    """Base for every exception raised by ``molexp.harness``."""


class ArtifactNotFoundError(HarnessError):
    """Raised when a lookup by ``artifact_id`` finds nothing."""


class EventSeqConflictError(HarnessError):
    """Raised when an attempted ``HarnessEvent.append`` violates the per-run
    ``(run_id, seq)`` uniqueness constraint.

    Wraps the underlying ``sqlite3.IntegrityError`` as ``__cause__``.
    """


class StageExecutionError(HarnessError):
    """Raised by :class:`molexp.harness.core.stage_runner.StageRunner` when a
    stage's ``run()`` raises.

    The original exception is chained as ``__cause__`` so callers retain the
    full traceback while still catching a typed harness error.
    """


class StagePersistedFailureError(StageExecutionError):
    """Raised by a Stage to signal "I have persisted a meaningful failure
    artifact (typically a validation_report) and now want the runner to
    fail the stage AND record the artifact's lineage events".

    Subclasses :class:`StageExecutionError` so callers using
    ``except StageExecutionError`` still catch it — including pipelines
    that call ``stage.run()`` directly without the
    :class:`StageRunner` wrapper. The runner catches this subclass first
    and emits ``artifact_created`` + ``derived_from`` edges for
    ``persisted_ref`` before re-raising, preserving the
    always-persist-then-raise contract for validators: the
    :class:`PlanValidationReport` is visible to downstream auditors even when
    the stage was strict and aborted the pipeline.

    Attributes:
        persisted_ref: The :class:`PlanArtifactRef` of the failure artifact
            (e.g. a parse-error PlanValidationReport) already written to the
            artifact store.
    """

    def __init__(self, persisted_ref: PlanArtifactRef, message: str) -> None:
        super().__init__(message)
        self.persisted_ref = persisted_ref


class ApprovalPendingError(StageExecutionError):
    """Raised by an approval gate that has no interactive approver and no
    stored grant: the pipeline **suspends** instead of granting.

    Subclasses :class:`StageExecutionError` so existing ``except
    StageExecutionError`` call sites still stop the pipeline — but callers
    that distinguish suspension from failure (task stores, CLI exit codes,
    record materialization) must catch this subclass FIRST: a suspension is
    "waiting for a human decision", never a failure to analyze.

    The pending request(s) are recorded in the run's
    :class:`~molexp.harness.store.approval_store.ApprovalStore` before this
    is raised, so the server inbox can list them and a later decision lets a
    ledger-resumed re-entry pass the gate.

    Attributes:
        requests: The :class:`ApprovalRequest`\\ s awaiting a decision.
        run_id: The harness run whose approval store holds them.
    """

    def __init__(self, requests: list[ApprovalRequest], run_id: str) -> None:
        intents = ", ".join(request.intent for request in requests)
        super().__init__(
            f"approval pending for run {run_id!r}: {len(requests)} request(s) "
            f"await a decision (intents: {intents}); decide via the approvals "
            "inbox (UI), an interactive TTY, or --yes, then re-run to resume"
        )
        self.requests = list(requests)
        self.run_id = run_id


class AgentResponseNotRegisteredError(HarnessError):
    """Raised by :class:`molexp.harness.gateways.stub.StubAgentGateway` when
    ``call()`` is invoked with an ``agent_name`` that has no registered
    canned response.

    Test-only; production code never hits this path.
    """


class CapabilityNotFoundError(HarnessError):
    """Raised by :class:`CapabilityRegistry.get` when ``capability_id`` has
    no registered entry."""


class CapabilityAlreadyRegisteredError(HarnessError):
    """Raised by :meth:`InMemoryCapabilityRegistry.register` when the
    capability's ``id`` is already in the registry."""


class CapabilityCallValidationError(HarnessError):
    """Raised by :meth:`CapabilityRegistry.validate_call` when the
    supplied ``parameters`` dict violates the capability's ``input_schema``
    contract (unknown capability, extra keys, missing required keys).

    Shallow contract — does NOT check value types against the schema."""


class CapabilityResolutionError(HarnessError):
    """Raised by :func:`molexp.harness.capability.resolve_callable` when a
    :class:`ToolCapability`'s ``callable_path`` cannot be dispatched to a real
    callable.

    Distinct from :class:`CapabilityNotFoundError` (a registry miss): the
    capability *is* registered, but its ``callable_path`` is ``None``/empty,
    names an unimportable module, references a missing attribute, or resolves
    to a non-callable object. There is no silent fallback — every dispatch
    failure surfaces as this typed error."""


class CapabilityGapError(StageExecutionError):
    """A plan step needs a molcrafts capability/API that molmcp cannot find.

    Not retryable by re-sampling the LLM: the operator must update the plan
    (pick an available capability) or update molpy/molmcp and re-index. Raised
    by repair loops when diagnosis reports a capability gap so attempts are
    not burned inventing missing symbols.

    Attributes:
        code: Stable machine code (e.g. ``unknown_capability``,
            ``api_symbol_missing``, ``molmcp_unavailable``).
        symbol: Optional dotted symbol / capability id.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "capability_missing",
        symbol: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.symbol = symbol


class UnhandledHighRiskOpError(HarnessError):
    """Raised by :meth:`molexp.harness.actions.ProposalExecutor.dispatch` when a
    granted :class:`~molexp.harness.schemas.change_proposal.ChangeProposal`'s
    ``proposed_change.op`` has no registered :class:`ChangeActionHandler`.

    A routing/config defect, not a proposal-execution outcome: it propagates out
    of ``dispatch`` (no ``tool_*`` event, no ``status="failed"`` record). There
    is no silent fallback — an unhandled high-risk op is always loud
    (integration.md §10 invariant #10)."""


class OutOfAffectedScopeError(HarnessError):
    """Raised by :func:`molexp.harness.actions.assert_within_affected_scope` when
    an action would touch an :class:`~molexp.harness.schemas.change_proposal.ObjectRef`
    absent from the proposal's ``affected_objects``.

    Enforces the §8.2 binding scope ("the ONLY objects Act may touch"). A handler
    that hits this is recorded by the executor as ``status="failed"`` (+ a
    ``tool_failed`` event), never a silent partial mutation."""


class ObjectRefResolutionError(HarnessError):
    """Raised by :func:`molexp.harness.actions.resolve_object_ref` when an
    :class:`~molexp.harness.schemas.change_proposal.ObjectRef` cannot be resolved
    to a live workspace entity.

    Covers a missing id (no run / experiment / folder / data asset with that id)
    and an unknown ``kind``. There is no silent ``None`` — an unresolvable ref is
    always loud (project no-fallback rule)."""
