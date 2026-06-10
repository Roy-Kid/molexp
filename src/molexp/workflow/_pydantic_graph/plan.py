"""ExecutionPlan — the lowered, executable workflow artifact.

:class:`ExecutionPlan` is what :meth:`WorkflowGraphCompiler.compile` emits and
what :class:`CompiledWorkflow.graph` carries. It is a **structural** plan over
the validated topology — out-edges, entry frontier, back-edges (cycles),
forward trigger sources per task, and the ``wf.parallel`` declarations — with
no live scheduling state. The engine (:mod:`.engine`) walks it per execution.

Values flow **on edges**: each completed task's recorded output rides the
trigger edges to its targets, and the engine launches a task exactly when all
of its live forward in-edges have fired and its declared ``depends_on`` values
are present. Deadlock is therefore detected structurally (an unsatisfiable
dependency, not a quiescence timer).

This module MUST NOT import ``pydantic_graph`` — the plan is plain data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .._graph_decl import ParallelDecl
    from ..types import OutEdges

# Pseudo-source for entry-frontier trigger edges. The engine fires one
# ``START → entry`` trigger per entry task when an execution begins, so entry
# readiness rides the same in-edge bookkeeping as every other task.
START = "__start__"


@dataclass(frozen=True)
class ExecutionPlan:
    """Frozen lowering artifact: the structural execution plan for one workflow.

    Attributes:
        name: Workflow name (observability).
        task_names: Registered task names in declaration order — the engine's
            deterministic iteration order.
        out_edges: ``task → OutEdges`` (``UnconditionalEdges`` fan-out or
            ``BranchEdges`` routes), as compiled by stages 1-4.
        entry_frontier: Tasks triggered at execution start.
        back_edges: Cycle-forming ``(src, tgt)`` edges (``wf.loop`` back-branch,
            self-loops). A back-edge trigger re-launches its target directly and
            never participates in forward in-edge coalescing.
        in_sources: ``task → forward trigger sources`` (:data:`START` for
            entries). A task is control-ready when every live source here has
            fired since its last launch.
        recurrent: Tasks on a control cycle (re-triggerable after completing).
            A non-recurrent branch permanently kills its non-chosen route
            edges when it routes; a recurrent one leaves them pending (a later
            iteration may fire them).
        parallels: All ``wf.parallel`` declarations.
        parallel_by_body: ``body task → ParallelDecl``.
        parallel_by_map_over: ``map_over task → ParallelDecl``.
    """

    name: str
    task_names: tuple[str, ...]
    out_edges: Mapping[str, OutEdges]
    entry_frontier: tuple[str, ...]
    back_edges: frozenset[tuple[str, str]]
    in_sources: Mapping[str, frozenset[str]]
    recurrent: frozenset[str] = frozenset()
    parallels: tuple[ParallelDecl, ...] = ()
    parallel_by_body: Mapping[str, ParallelDecl] = field(default_factory=dict)
    parallel_by_map_over: Mapping[str, ParallelDecl] = field(default_factory=dict)


__all__ = ["START", "ExecutionPlan"]
