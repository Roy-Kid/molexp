"""Graph-structure analysis for workflow lowering.

Pure functions over the compiled out-edge structure used by
:class:`molexp.workflow._pydantic_graph.compiler.WorkflowGraphCompiler`:
target enumeration, back-edge (cycle) detection, and forward-indegree
counting. No ``pydantic_graph`` import and no compiler state — just the
lowered edge maps — so they live apart from the (large) compiler module.
"""

from __future__ import annotations

from collections import defaultdict

from .._graph_decl import ParallelDecl, WorkflowTopology
from ..types import BranchEdges, OutEdges, UnconditionalEdges
from .node import END_TARGET


def iter_targets(edge_set: OutEdges) -> list[str]:
    """All edge targets of *edge_set*, excluding ``_end``."""
    if isinstance(edge_set, UnconditionalEdges):
        return [t for t in edge_set.targets if t != END_TARGET]
    if isinstance(edge_set, BranchEdges):
        return [t for t in edge_set.routes.values() if t != END_TARGET]
    return []


def compute_back_edges(
    out_edges: dict[str, OutEdges],
    entry_frontier: tuple[str, ...],
    parallel_decls: dict[str, ParallelDecl],
) -> set[tuple[str, str]]:
    """Find cycle-forming edges via DFS over the lowered out-edge graph.

    An edge ``(src, tgt)`` is a back-edge iff ``tgt`` is on the current DFS
    recursion stack when the edge is traversed (the classic grey-node test).
    These edges form loops (``wf.loop`` back-branch, self-loops) and must
    re-enter the target Step directly rather than through a coalescing Join.
    Parallel body wiring is owned by the parallel primitive and excluded from
    this walk.
    """
    back_edges: set[tuple[str, str]] = set()
    visited: set[str] = set()
    on_stack: set[str] = set()

    def dfs(node: str) -> None:
        visited.add(node)
        on_stack.add(node)
        if node not in parallel_decls:
            for tgt in iter_targets(out_edges.get(node, UnconditionalEdges(targets=()))):
                if tgt in parallel_decls:
                    continue
                if tgt in on_stack:
                    back_edges.add((node, tgt))
                elif tgt not in visited:
                    dfs(tgt)
        on_stack.discard(node)

    for entry in entry_frontier:
        if entry not in visited:
            dfs(entry)
    return back_edges


def compute_recurrent(
    out_edges: dict[str, OutEdges],
    parallel_decls: dict[str, ParallelDecl],
) -> frozenset[str]:
    """Tasks lying on a control cycle — i.e. re-triggerable after completing.

    A branch task NOT on a cycle completes exactly once, so its non-chosen
    route edges are permanently dead the moment it routes (structural death
    propagates). A branch ON a cycle (``wf.loop`` until, self-loop) may fire a
    different label on a later iteration, so its non-chosen edges stay live.
    Computed as strongly-connected reachability: ``t`` is recurrent iff ``t``
    is reachable from one of its own successors.
    """
    successors: dict[str, list[str]] = {}
    for src, edge_set in out_edges.items():
        if src in parallel_decls:
            continue
        successors[src] = [t for t in iter_targets(edge_set) if t not in parallel_decls]

    recurrent: set[str] = set()
    for start in successors:
        stack = list(successors.get(start, ()))
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node == start:
                recurrent.add(start)
                break
            if node in seen:
                continue
            seen.add(node)
            stack.extend(successors.get(node, ()))
    return frozenset(recurrent)


def compute_in_sources(
    spec: WorkflowTopology,
    out_edges: dict[str, OutEdges],
    entry_frontier: tuple[str, ...],
    parallel_decls: dict[str, ParallelDecl],
    back_edges: set[tuple[str, str]],
) -> dict[str, frozenset[str]]:
    """Collect each task's *forward* trigger sources across the edge structure.

    Entry tasks get the :data:`~.plan.START` pseudo-source. Back-edges
    (cycles) are excluded — a back-edge trigger re-launches its target
    directly and never participates in forward coalescing. ``wf.parallel``
    body wiring is owned by the parallel primitive: the body task has no
    forward sources (the engine fans it out from ``map_over`` directly) and
    the join task gains the body as its source. End targets never count.
    """
    from .plan import START

    in_sources: defaultdict[str, set[str]] = defaultdict(set)
    for entry in entry_frontier:
        if entry not in parallel_decls:
            in_sources[entry].add(START)
    for src, edge_set in out_edges.items():
        if src in parallel_decls:
            continue  # body's out-edges fire after the fan-out publishes
        for tgt in iter_targets(edge_set):
            if tgt in parallel_decls or (src, tgt) in back_edges:
                continue
            in_sources[tgt].add(src)
    # The published fan-out → join edge is the join's incoming trigger.
    for par in spec._parallels:
        in_sources[par.join].add(par.body)
    return {name: frozenset(sources) for name, sources in in_sources.items()}
