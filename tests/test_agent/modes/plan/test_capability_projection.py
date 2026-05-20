"""Capability-graph projection tests (ac-003).

``capability_projection`` turns a flat ``ProbeResult`` (drafted needs +
evidence batch) into a typed ``CapabilityGraph`` whose nodes carry an
evidence state, confidence, api refs, and usage limits, with
alternative / depends_on edges.
"""

from __future__ import annotations

from molexp.agent.modes._planning import (
    CapabilityEdgeKind,
    CapabilityGraph,
    EvidenceState,
)
from molexp.agent.modes.plan.capability_evidence import (
    CapabilityEvidenceBatch,
    DraftedNeed,
)
from molexp.agent.modes.plan.capability_projection import capability_projection
from molexp.agent.modes.plan.protocols import ProbeResult

from .conftest import make_probe_result


def test_projection_produces_one_node_per_need() -> None:
    graph = capability_projection(make_probe_result())
    assert isinstance(graph, CapabilityGraph)
    assert {node.id for node in graph.nodes} == {"build_system", "run_md"}


def test_projection_marks_evidenced_nodes() -> None:
    graph = capability_projection(make_probe_result())
    build = graph.node_by_id("build_system")
    assert build is not None
    assert build.evidence_state is EvidenceState.evidenced
    assert build.confidence == 0.95
    assert "molpy.System" in build.api_refs


def test_projection_carries_usage_limits() -> None:
    graph = capability_projection(make_probe_result())
    run_md = graph.node_by_id("run_md")
    assert run_md is not None
    assert "requires a LAMMPS install" in run_md.usage_limits
    assert run_md.needs_user_confirmation is True


def test_projection_emits_depends_on_edges() -> None:
    graph = capability_projection(make_probe_result())
    depends = [e for e in graph.edges if e.kind is CapabilityEdgeKind.depends_on]
    assert any(e.source == "run_md" and e.target == "build_system" for e in depends)


def test_projection_emits_alternative_edges() -> None:
    result = ProbeResult(
        drafted_needs=(
            DraftedNeed(
                need_id="fast_path",
                capability="quick approximation",
                rationale="",
                api_refs=("pkg.fast",),
                depends_on=(),
                alternatives=("slow_path",),
                needs_user_confirmation=False,
            ),
            DraftedNeed(
                need_id="slow_path",
                capability="accurate computation",
                rationale="",
                api_refs=("pkg.slow",),
                depends_on=(),
                alternatives=(),
                needs_user_confirmation=False,
            ),
        ),
        evidence=CapabilityEvidenceBatch(items=(), missing_refs=()),
    )
    graph = capability_projection(result)
    alts = [e for e in graph.edges if e.kind is CapabilityEdgeKind.alternative]
    assert any(e.source == "fast_path" and e.target == "slow_path" for e in alts)


def test_projection_marks_need_without_evidence_as_missing() -> None:
    result = ProbeResult(
        drafted_needs=(
            DraftedNeed(
                need_id="unknown",
                capability="something undiscoverable",
                rationale="",
                api_refs=("ghost.api",),
                depends_on=(),
                alternatives=(),
                needs_user_confirmation=False,
            ),
        ),
        evidence=CapabilityEvidenceBatch(items=(), missing_refs=("ghost.api",)),
    )
    graph = capability_projection(result)
    node = graph.node_by_id("unknown")
    assert node is not None
    assert node.evidence_state is EvidenceState.missing
    assert node.confidence == 0.0
