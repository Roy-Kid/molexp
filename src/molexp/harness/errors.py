"""Typed exceptions for ``molexp.harness``.

A single ``HarnessError`` base lets call sites catch every harness-raised
exception with one ``except`` clause; the leaf subclasses are what the
harness raises in practice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.harness.schemas import ArtifactRef

__all__ = [
    "AgentResponseNotRegisteredError",
    "ArtifactNotFoundError",
    "CapabilityAlreadyRegisteredError",
    "CapabilityCallValidationError",
    "CapabilityNotFoundError",
    "CapabilityResolutionError",
    "EventSeqConflictError",
    "HarnessError",
    "StageExecutionError",
    "StagePersistedFailureError",
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
    :class:`ValidationReport` is visible to downstream auditors even when
    the stage was strict and aborted the pipeline.

    Attributes:
        persisted_ref: The :class:`ArtifactRef` of the failure artifact
            (e.g. a parse-error ValidationReport) already written to the
            artifact store.
    """

    def __init__(self, persisted_ref: ArtifactRef, message: str) -> None:
        super().__init__(message)
        self.persisted_ref = persisted_ref


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
