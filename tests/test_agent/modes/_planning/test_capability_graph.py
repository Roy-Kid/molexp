"""RED-phase tests for the capability-graph cluster of
``molexp.agent.modes._planning``.

The package does not exist yet; these tests fail at collection until the
implementation lands.

Covers, per the testing rules:

- Basics  — full valid construction + ``model_dump(mode="json")`` round-trip.
- Edge cases — ``extra="forbid"`` rejects an unknown field;
  out-of-range ``confidence`` (below 0.0 / above 1.0) is rejected.
- Immutability — ``frozen=True`` rejects attribute assignment.
- Logic — ``node_by_id`` hit/miss; ``alternatives_of`` follows
  ``alternative`` edges only.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from molexp.agent.modes._planning import (
    CapabilityEdge,
    CapabilityEdgeKind,
    CapabilityGraph,
    CapabilityNode,
    EvidenceState,
)

# --------------------------------------------------------------------------
# fixtures (hand-built; no LLM, no I/O)
# --------------------------------------------------------------------------


def _node(node_id: str, *, confidence: float = 0.8) -> CapabilityNode:
    return CapabilityNode(
        id=node_id,
        capability=f"capability::{node_id}",
        evidence_state=EvidenceState.evidenced,
        confidence=confidence,
        api_refs=(f"molexp.{node_id}",),
        usage_limits=("rate-limited",),
        needs_user_confirmation=False,
    )


def _graph() -> CapabilityGraph:
    # primary -alternative-> backup ; primary -depends_on-> base
    return CapabilityGraph(
        nodes=(_node("primary"), _node("backup"), _node("base")),
        edges=(
            CapabilityEdge(source="primary", target="backup", kind=CapabilityEdgeKind.alternative),
            CapabilityEdge(source="primary", target="base", kind=CapabilityEdgeKind.depends_on),
        ),
    )


# --------------------------------------------------------------------------
# basics — enums
# --------------------------------------------------------------------------


def test_evidence_state_members() -> None:
    assert {m.value for m in EvidenceState} == {
        "unevidenced",
        "evidenced",
        "missing",
        "assumed",
    }


def test_capability_edge_kind_members() -> None:
    assert {m.value for m in CapabilityEdgeKind} == {"alternative", "depends_on"}


# --------------------------------------------------------------------------
# basics — node / edge / graph construction
# --------------------------------------------------------------------------


def test_capability_node_construction() -> None:
    node = _node("primary")
    assert node.id == "primary"
    assert node.evidence_state is EvidenceState.evidenced
    assert node.confidence == 0.8
    assert node.needs_user_confirmation is False


def test_capability_edge_construction() -> None:
    edge = CapabilityEdge(source="a", target="b", kind=CapabilityEdgeKind.depends_on)
    assert edge.source == "a"
    assert edge.target == "b"
    assert edge.kind is CapabilityEdgeKind.depends_on


def test_capability_graph_construction() -> None:
    graph = _graph()
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2


# --------------------------------------------------------------------------
# basics — confidence bounds accepted
# --------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_capability_node_accepts_in_range_confidence(confidence: float) -> None:
    node = _node("primary", confidence=confidence)
    assert node.confidence == confidence


# --------------------------------------------------------------------------
# basics — JSON round-trip
# --------------------------------------------------------------------------


def test_capability_graph_json_round_trip() -> None:
    graph = _graph()
    dumped = graph.model_dump(mode="json")
    restored = CapabilityGraph.model_validate(dumped)
    assert restored == graph


def test_capability_graph_json_dump_is_jsonable() -> None:
    dumped = _graph().model_dump(mode="json")
    assert isinstance(dumped["nodes"], list)
    assert dumped["edges"][0]["kind"] == "alternative"
    assert dumped["nodes"][0]["evidence_state"] == "evidenced"


# --------------------------------------------------------------------------
# edge cases — extra="forbid"
# --------------------------------------------------------------------------


def test_capability_node_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        CapabilityNode(
            id="x",
            capability="c",
            evidence_state=EvidenceState.assumed,
            confidence=0.5,
            api_refs=(),
            usage_limits=(),
            needs_user_confirmation=False,
            score=1,  # type: ignore[call-arg]
        )


def test_capability_edge_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        CapabilityEdge(
            source="a",
            target="b",
            kind=CapabilityEdgeKind.alternative,
            weight=2,  # type: ignore[call-arg]
        )


def test_capability_graph_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        CapabilityGraph(nodes=(), edges=(), label="g")  # type: ignore[call-arg]


# --------------------------------------------------------------------------
# edge cases — confidence out of range
# --------------------------------------------------------------------------


def test_capability_node_rejects_confidence_above_one() -> None:
    with pytest.raises(ValidationError):
        _node("primary", confidence=1.01)


def test_capability_node_rejects_confidence_below_zero() -> None:
    with pytest.raises(ValidationError):
        _node("primary", confidence=-0.01)


# --------------------------------------------------------------------------
# immutability — frozen=True
# --------------------------------------------------------------------------


def test_capability_node_is_frozen() -> None:
    node = _node("primary")
    with pytest.raises(ValidationError):
        node.confidence = 0.1  # type: ignore[misc]


def test_capability_edge_is_frozen() -> None:
    edge = CapabilityEdge(source="a", target="b", kind=CapabilityEdgeKind.alternative)
    with pytest.raises(ValidationError):
        edge.target = "c"  # type: ignore[misc]


def test_capability_graph_is_frozen() -> None:
    graph = _graph()
    with pytest.raises(ValidationError):
        graph.nodes = ()  # type: ignore[misc]


# --------------------------------------------------------------------------
# logic — node_by_id
# --------------------------------------------------------------------------


def test_node_by_id_returns_matching_node() -> None:
    graph = _graph()
    found = graph.node_by_id("backup")
    assert found is not None
    assert found.id == "backup"


def test_node_by_id_returns_none_for_unknown() -> None:
    assert _graph().node_by_id("does-not-exist") is None


# --------------------------------------------------------------------------
# logic — alternatives_of
# --------------------------------------------------------------------------


def test_alternatives_of_follows_alternative_edge() -> None:
    alternatives = _graph().alternatives_of("primary")
    assert tuple(n.id for n in alternatives) == ("backup",)


def test_alternatives_of_ignores_depends_on_edge() -> None:
    # "base" is reached from "primary" via a depends_on edge, not alternative.
    alternatives = _graph().alternatives_of("primary")
    assert "base" not in {n.id for n in alternatives}


def test_alternatives_of_empty_for_node_without_alternatives() -> None:
    assert _graph().alternatives_of("base") == ()
