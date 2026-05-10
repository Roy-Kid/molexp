"""Domain exceptions for the PlanMode pipeline.

Three exceptions are exposed:

- :class:`SkeletonCompileError` — ``GenerateWorkflowSkeleton`` raises
  this when the Python source it emits fails :func:`compile`.
- :class:`CapabilityDiscoveryRequired` — raised when a downstream node
  asks for capability discovery but no probe is configured (or the
  probe declined the request). The ``planmode-review-repair-loop``
  driver intercepts this and re-runs ``DraftCapabilityNeeds`` +
  ``DiscoverCapabilities``.
- :class:`UnevidencedApiReference` — raised by codegen nodes after
  :func:`~molexp.agent.modes.plan.capability.validate_codegen_evidence`
  reports a non-empty miss set, or when the LLM's ``evidence_refs``
  schema field disagrees with the source's
  ``__capability_evidence__`` literal. The repair loop re-runs
  ``DiscoverCapabilities`` on the first miss and escalates to
  ``DraftCapabilityNeeds`` + ``DiscoverCapabilities`` on the second.

All three subclass :class:`RuntimeError` — the failure mode is a
code-generation / configuration defect, not a programmer error.
"""

from __future__ import annotations

__all__ = [
    "CapabilityDiscoveryRequired",
    "SkeletonCompileError",
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


class CapabilityDiscoveryRequired(RuntimeError):
    """Capability discovery is needed but cannot proceed.

    Raised by ``NullCapabilityProbe.discover`` when its input report
    sets ``discovery_required=True`` (no probe configured, molmcp
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


class UnevidencedApiReference(RuntimeError):
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
