"""Flat capability-evidence types — the raw shape a ``CapabilityProbe`` returns.

These are the *flat* probe output models, deliberately distinct from the
typed :class:`~molexp.agent.modes._planning.CapabilityGraph`: a probe
returns drafted needs plus an evidence batch, and
:func:`~molexp.agent.modes.plan.capability_projection.capability_projection`
folds them into the typed graph. Keeping the two apart lets the probe
stay a simple flat producer while the graph carries the structure
PlanMode's plan synthesis selects an approach *from*.

Pure frozen-pydantic data models; no LLM, no I/O.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CapabilityEvidenceBatch",
    "CapabilityEvidenceItem",
    "DraftedNeed",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class DraftedNeed(BaseModel):
    """One capability the plan plausibly needs, drafted before discovery.

    Drafted by the probe's no-tool needs agent from the typed
    :class:`~molexp.agent.modes._planning.IntentSpec`. Each becomes a
    :class:`~molexp.agent.modes._planning.CapabilityNode` after
    projection.

    Attributes:
        need_id: Stable identifier — becomes the ``CapabilityNode.id``.
        capability: One-phrase description of the capability needed.
        rationale: One sentence on why the plan needs it.
        api_refs: Candidate project API references for the capability.
        depends_on: ``need_id``s this need depends on.
        alternatives: ``need_id``s that are interchangeable alternatives.
        needs_user_confirmation: Whether the user must confirm its use.
    """

    model_config = _FROZEN

    need_id: str
    capability: str
    rationale: str = ""
    api_refs: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    needs_user_confirmation: bool = False


class CapabilityEvidenceItem(BaseModel):
    """One resolved piece of capability evidence from source introspection.

    Attributes:
        need_id: The :class:`DraftedNeed` this evidence resolves.
        api_ref: Fully-qualified reference (``f"{module}.{symbol}"``).
        module: The module the symbol lives in.
        symbol: The symbol name.
        kind: ``class`` / ``callable`` / ``module`` / ``constant`` / ….
        signature: The symbol's signature, if applicable.
        doc_summary: A one-line docstring summary.
        confidence: Confidence in the match, in ``[0.0, 1.0]``.
        usage_notes: Caveats / limits on using the symbol.
    """

    model_config = _FROZEN

    need_id: str
    api_ref: str
    module: str = ""
    symbol: str = ""
    kind: str = ""
    signature: str = ""
    doc_summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    usage_notes: tuple[str, ...] = ()


class CapabilityEvidenceBatch(BaseModel):
    """The flat evidence a probe gathers for a set of drafted needs.

    Attributes:
        items: The resolved evidence rows.
        missing_refs: ``api_ref``s the probe could not resolve.
    """

    model_config = _FROZEN

    items: tuple[CapabilityEvidenceItem, ...] = ()
    missing_refs: tuple[str, ...] = ()

    def items_for(self, need_id: str) -> tuple[CapabilityEvidenceItem, ...]:
        """Return every evidence item resolving ``need_id``."""
        return tuple(item for item in self.items if item.need_id == need_id)
