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


def compute_indegree(
    spec: WorkflowTopology,
    out_edges: dict[str, OutEdges],
    entry_frontier: tuple[str, ...],
    parallel_decls: dict[str, ParallelDecl],
    back_edges: set[tuple[str, str]],
) -> dict[str, int]:
    """Count *forward* incoming edges per task across the edge structure.

    Entry edges from ``start_node`` count as incoming. Back-edges (cycles) are
    excluded — they re-enter the target Step directly and must not synthesise a
    coalescing Join. ``wf.parallel`` targets the join task via the collector
    Step (one incoming edge); the body / map_over / join wiring is owned by the
    parallel and excluded here. End targets never count.
    """
    indegree: dict[str, int] = defaultdict(int)
    for entry in entry_frontier:
        indegree[entry] += 1
    for src, edge_set in out_edges.items():
        if src in parallel_decls:
            continue  # body's out-edges owned by the parallel
        for tgt in iter_targets(edge_set):
            if tgt in parallel_decls or (src, tgt) in back_edges:
                continue
            indegree[tgt] += 1
    # The collector → join edge is one incoming edge into each parallel join.
    for par in spec._parallels:
        indegree[par.join] += 1
    return dict(indegree)
