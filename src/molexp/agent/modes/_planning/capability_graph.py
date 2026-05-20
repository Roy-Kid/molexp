"""Capability-graph cluster — what the system can do, with evidence.

A typed graph of capabilities replacing the flat evidence batch: each
:class:`CapabilityNode` carries an evidence state, a confidence score,
usage limits, and a needs-user-confirmation flag; :class:`CapabilityEdge`
records alternative-path and dependency relations. Plan synthesis selects
an approach *from* this graph.

Pure frozen-pydantic data models; no LLM, no I/O.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EvidenceState(StrEnum):
    """How well a capability is backed by evidence."""

    unevidenced = "unevidenced"
    evidenced = "evidenced"
    missing = "missing"
    assumed = "assumed"


class CapabilityEdgeKind(StrEnum):
    """The relation a :class:`CapabilityEdge` records."""

    alternative = "alternative"
    depends_on = "depends_on"


class CapabilityNode(BaseModel):
    """One capability the system may use, with its evidence.

    Attributes:
        id: Stable identifier, referenced by ``PlanStep.capability_id``.
        capability: Human-readable description of the capability.
        evidence_state: How well the capability is evidenced.
        confidence: Confidence in the evidence, in ``[0.0, 1.0]``.
        api_refs: Concrete API references backing the capability.
        usage_limits: Known limits / caveats on using it.
        needs_user_confirmation: Whether the user must confirm its use.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    capability: str
    evidence_state: EvidenceState
    confidence: float = Field(ge=0.0, le=1.0)
    api_refs: tuple[str, ...]
    usage_limits: tuple[str, ...]
    needs_user_confirmation: bool


class CapabilityEdge(BaseModel):
    """A directed relation between two capability nodes.

    Attributes:
        source: ``id`` of the source node.
        target: ``id`` of the target node.
        kind: The relation kind (alternative path or dependency).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    target: str
    kind: CapabilityEdgeKind


class CapabilityGraph(BaseModel):
    """The typed capability graph — nodes plus their relations.

    Attributes:
        nodes: The capability nodes.
        edges: The relations between nodes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: tuple[CapabilityNode, ...]
    edges: tuple[CapabilityEdge, ...]

    def node_by_id(self, node_id: str) -> CapabilityNode | None:
        """Return the node with ``id == node_id``, or ``None``."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def alternatives_of(self, node_id: str) -> tuple[CapabilityNode, ...]:
        """Return nodes reachable from ``node_id`` via an alternative edge.

        Follows ``alternative``-kind edges only; ``depends_on`` edges are
        ignored.
        """
        targets = [
            edge.target
            for edge in self.edges
            if edge.kind is CapabilityEdgeKind.alternative and edge.source == node_id
        ]
        resolved = (self.node_by_id(target) for target in targets)
        return tuple(node for node in resolved if node is not None)
