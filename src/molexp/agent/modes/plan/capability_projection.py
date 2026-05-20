"""Project a flat ``ProbeResult`` into a typed ``CapabilityGraph``.

``ExploreCapabilities`` calls the :class:`CapabilityProbe`, then folds
the flat output (drafted needs + evidence batch) into the typed
:class:`~molexp.agent.modes._planning.CapabilityGraph` plan synthesis
selects an approach from. One :class:`DraftedNeed` becomes one
:class:`CapabilityNode`; ``depends_on`` / ``alternatives`` become typed
:class:`CapabilityEdge`\\ s.

Pure functions; no LLM, no I/O.
"""

from __future__ import annotations

from molexp.agent.modes._planning import (
    CapabilityEdge,
    CapabilityEdgeKind,
    CapabilityGraph,
    CapabilityNode,
    EvidenceState,
)
from molexp.agent.modes.plan.capability_evidence import (
    CapabilityEvidenceBatch,
    DraftedNeed,
)
from molexp.agent.modes.plan.protocols import ProbeResult

__all__ = ["capability_projection"]


def capability_projection(result: ProbeResult) -> CapabilityGraph:
    """Fold a flat :class:`ProbeResult` into a typed :class:`CapabilityGraph`.

    Each :class:`DraftedNeed` becomes one :class:`CapabilityNode` whose
    evidence state is derived from the matching
    :class:`~molexp.agent.modes.plan.capability_evidence.CapabilityEvidenceItem`\\ s:

    * at least one evidence item → :data:`EvidenceState.evidenced`;
    * the need's api refs are all in ``missing_refs`` → :data:`EvidenceState.missing`;
    * no evidence and not flagged missing → :data:`EvidenceState.unevidenced`.

    Confidence is the mean of the matched evidence items' confidence (or
    ``0.0`` with no evidence); usage limits are the union of every
    matched item's ``usage_notes``.
    """
    nodes = tuple(_project_node(need, result.evidence) for need in result.drafted_needs)
    edges = _project_edges(result.drafted_needs)
    return CapabilityGraph(nodes=nodes, edges=edges)


def _project_node(need: DraftedNeed, evidence: CapabilityEvidenceBatch) -> CapabilityNode:
    """Build one typed :class:`CapabilityNode` for a drafted need."""
    items = evidence.items_for(need.need_id)
    state = _evidence_state(need, items, evidence)
    confidence = sum(item.confidence for item in items) / len(items) if items else 0.0
    api_refs = _resolve_api_refs(need, items)
    usage_limits = _collect_usage_limits(items)
    return CapabilityNode(
        id=need.need_id,
        capability=need.capability,
        evidence_state=state,
        confidence=confidence,
        api_refs=api_refs,
        usage_limits=usage_limits,
        needs_user_confirmation=need.needs_user_confirmation,
    )


def _evidence_state(
    need: DraftedNeed,
    items: tuple[object, ...],
    evidence: CapabilityEvidenceBatch,
) -> EvidenceState:
    """Classify a need's evidence state from its matched items."""
    if items:
        return EvidenceState.evidenced
    if need.api_refs and all(ref in evidence.missing_refs for ref in need.api_refs):
        return EvidenceState.missing
    return EvidenceState.unevidenced


def _resolve_api_refs(need: DraftedNeed, items: tuple[object, ...]) -> tuple[str, ...]:
    """Prefer evidenced api refs; fall back to the drafted candidates."""
    evidenced = tuple(ref for ref in (getattr(item, "api_ref", "") for item in items) if ref)
    return evidenced or need.api_refs


def _collect_usage_limits(items: tuple[object, ...]) -> tuple[str, ...]:
    """Union every matched evidence item's ``usage_notes``, order-preserving."""
    seen: list[str] = []
    for item in items:
        for note in getattr(item, "usage_notes", ()):
            if note not in seen:
                seen.append(note)
    return tuple(seen)


def _project_edges(needs: tuple[DraftedNeed, ...]) -> tuple[CapabilityEdge, ...]:
    """Project ``depends_on`` / ``alternatives`` into typed edges.

    An edge is emitted only when its target is itself a drafted need —
    dangling references are silently dropped (the preflight catches
    unsatisfiable plans separately).
    """
    known = {need.need_id for need in needs}
    edges: list[CapabilityEdge] = []
    for need in needs:
        for dep in need.depends_on:
            if dep in known:
                edges.append(
                    CapabilityEdge(
                        source=need.need_id,
                        target=dep,
                        kind=CapabilityEdgeKind.depends_on,
                    )
                )
        for alt in need.alternatives:
            if alt in known:
                edges.append(
                    CapabilityEdge(
                        source=need.need_id,
                        target=alt,
                        kind=CapabilityEdgeKind.alternative,
                    )
                )
    return tuple(edges)
