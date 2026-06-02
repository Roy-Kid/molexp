"""``WorkflowGraphIR`` — the full compiled-graph intermediate representation.

This is the workflow layer's *own* IR: a frozen, JSON-serializable snapshot
of everything a compiled :class:`~molexp.workflow.spec.Workflow` holds — its
tasks and dependencies plus the complete control-flow topology (entries,
control edges, branch routes, loops, parallel fan-outs). It is produced by
:meth:`Workflow.to_ir` after :meth:`WorkflowBuilder.build`.

It is deliberately distinct from two neighbours:

- The JSON *wire* IR produced by :meth:`Workflow.to_dict`
  (``schema/workflow.json`` — ``task_configs`` + ``links``) is the
  server/agent contract and is intentionally DAG-only; it rejects control
  flow. ``WorkflowGraphIR`` is a superset that *can* represent branches,
  loops, and parallels, so it is the right surface for diagrams and review.
- :class:`molexp.harness.schemas.workflow_ir.WorkflowIR` is the higher-level
  *scientific-intent* IR (what the experiment wants to compute). This IR is
  the *executable-graph* layer (how the compiled workflow is wired).

All models are ``frozen`` pydantic, matching the layer convention that pure
data types are immutable. Edge collections are tuples so the IR round-trips
through ``model_dump(mode="json")`` / ``model_validate`` unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from .._typing import JSONValue

if TYPE_CHECKING:
    from .spec import Workflow

__all__ = [
    "GraphLoopIR",
    "GraphParallelIR",
    "GraphTaskIR",
    "WorkflowGraphIR",
    "build_workflow_graph_ir",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class GraphTaskIR(BaseModel):
    """One task or actor node in a :class:`WorkflowGraphIR`."""

    model_config = _FROZEN

    name: str
    task_type: str | None = None
    depends_on: tuple[str, ...] = ()
    is_actor: bool = False
    config: dict[str, JSONValue] = Field(default_factory=dict)


class GraphLoopIR(BaseModel):
    """A ``wf.loop`` declaration: ``body`` repeats until ``until`` exits."""

    model_config = _FROZEN

    body: tuple[str, ...]
    until: str
    max_iters: int
    on_exit: str


class GraphParallelIR(BaseModel):
    """A ``wf.parallel`` declaration: ``body`` runs once per ``map_over`` element."""

    model_config = _FROZEN

    map_over: str
    body: str
    join: str
    max_concurrency: int


class WorkflowGraphIR(BaseModel):
    """Frozen, JSON-serializable snapshot of a compiled workflow's full graph.

    Captures the complete topology a :class:`~molexp.workflow.spec.Workflow`
    carries — tasks + dependencies and every control-flow primitive. Unlike
    the wire IR (:meth:`Workflow.to_dict`), it never raises on control flow
    and never requires ``task_type`` slugs, so decorator-defined workflows
    serialize cleanly too.
    """

    model_config = _FROZEN

    name: str
    workflow_id: str
    version: str = "0"
    mode: str = "batch"
    tasks: tuple[GraphTaskIR, ...] = ()
    entries: tuple[str, ...] = ()
    control_edges: tuple[tuple[str, str], ...] = ()
    branch_edges: tuple[tuple[str, str, str], ...] = ()
    loops: tuple[GraphLoopIR, ...] = ()
    parallels: tuple[GraphParallelIR, ...] = ()

    def to_mermaid(self) -> str:
        """Render this IR as a Mermaid ``flowchart`` (see :mod:`.mermaid`)."""
        from .mermaid import render_workflow_mermaid

        return render_workflow_mermaid(self)


def build_workflow_graph_ir(spec: Workflow) -> WorkflowGraphIR:
    """Build the full :class:`WorkflowGraphIR` from a compiled :class:`Workflow`.

    Reads the spec's frozen topology directly; every task is included
    regardless of whether it carries a registry ``task_type`` slug.
    """
    tasks = tuple(
        GraphTaskIR(
            name=t.name,
            task_type=t.task_type,
            depends_on=tuple(t.depends_on),
            is_actor=t.is_actor,
            config=dict(t.config) if t.config else {},
        )
        for t in spec._tasks
    )
    loops = tuple(
        GraphLoopIR(
            body=tuple(loop.body),
            until=loop.until,
            max_iters=loop.max_iters,
            on_exit=loop.on_exit,
        )
        for loop in spec._loops
    )
    parallels = tuple(
        GraphParallelIR(
            map_over=p.map_over, body=p.body, join=p.join, max_concurrency=p.max_concurrency
        )
        for p in spec._parallels
    )
    return WorkflowGraphIR(
        name=spec.name,
        workflow_id=spec.workflow_id,
        version=spec.version_label,
        mode=spec._mode,
        tasks=tasks,
        entries=tuple(spec._entries),
        control_edges=tuple(spec._control_edges),
        branch_edges=tuple(spec._branch_edges),
        loops=loops,
        parallels=parallels,
    )
