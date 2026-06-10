"""``WorkflowGraphIR`` — the full compiled-graph intermediate representation.

This is the workflow layer's *own* IR: a frozen, JSON-serializable snapshot
of everything a compiled :class:`~molexp.workflow.spec.Workflow` holds — its
tasks and dependencies plus the complete control-flow topology (entries,
control edges, branch routes, loops, parallel fan-outs). It is produced by
:meth:`CompiledWorkflow.to_ir` after :meth:`WorkflowCompiler.compile`.

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

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from .._typing import JSONValue
from .snapshot import task_config_of

if TYPE_CHECKING:
    from .compiled import CompiledWorkflow as Workflow

__all__ = [
    "EdgeKind",
    "GraphEdgeIR",
    "GraphLoopIR",
    "GraphNodePosition",
    "GraphParallelIR",
    "GraphTaskIR",
    "WorkflowGraphIR",
    "build_workflow_graph_ir",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")

#: The five edge kinds a free-layout workflow edge can carry. ``data`` is a
#: dependency edge (``depends_on``); ``control`` an unconditional control edge;
#: ``branch`` a label-routed edge (carries ``condition``); ``loop`` / ``parallel``
#: the synthesized edges of a ``wf.loop`` / ``wf.parallel`` declaration.
EdgeKind = Literal["data", "control", "branch", "loop", "parallel"]


class GraphNodePosition(BaseModel):
    """Editor-canvas coordinate for a node (free-layout graph).

    Pure UI metadata — it round-trips through the IR so an edited canvas can
    persist node placement, but it never enters the
    :class:`~molexp.workflow.snapshot.TaskSnapshot` content hash, so moving a
    node never invalidates the content-addressed cache.
    """

    model_config = _FROZEN

    x: float
    y: float


class GraphEdgeIR(BaseModel):
    """One typed edge in a :class:`WorkflowGraphIR`'s unified edge set."""

    model_config = _FROZEN

    source: str
    target: str
    kind: EdgeKind
    condition: str | None = None


class GraphTaskIR(BaseModel):
    """One task or actor node in a :class:`WorkflowGraphIR`."""

    model_config = _FROZEN

    name: str
    task_type: str | None = None
    depends_on: tuple[str, ...] = ()
    is_actor: bool = False
    config: dict[str, JSONValue] = Field(default_factory=dict)
    position: GraphNodePosition | None = None
    #: When this node is a :class:`~molexp.workflow.SubWorkflow`, the full graph
    #: IR of the embedded inner workflow — so a UI can render the node with a
    #: distinct badge and drill into the inner topology (read-only). ``None`` for
    #: ordinary task/actor nodes.
    subworkflow: WorkflowGraphIR | None = None


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
    #: Unified, ``kind``-tagged edge set spanning all five control-flow
    #: shapes — the surface a free-layout canvas renders. Derived from the
    #: separate collections above by :func:`build_workflow_graph_ir`.
    edges: tuple[GraphEdgeIR, ...] = ()

    def to_mermaid(self) -> str:
        """Render this IR as a Mermaid ``flowchart`` (see :mod:`.mermaid`)."""
        from .mermaid import render_workflow_mermaid

        return render_workflow_mermaid(self)


def build_workflow_graph_ir(spec: Workflow) -> WorkflowGraphIR:
    """Build the full :class:`WorkflowGraphIR` from a compiled :class:`Workflow`.

    Reads the spec's frozen topology directly; every task is included
    regardless of whether it carries a registry ``task_type`` slug.
    """
    from .subworkflow import SubWorkflow

    tasks = tuple(
        GraphTaskIR(
            name=t.name,
            task_type=t.task_type,
            depends_on=tuple(t.depends_on),
            is_actor=t.is_actor,
            config=_node_config(t.fn_or_class),
            position=_position_of(t),
            subworkflow=(
                t.fn_or_class.inner.to_graph_ir()
                if isinstance(t.fn_or_class, SubWorkflow)
                else None
            ),
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
        edges=_build_edges(spec),
    )


def _node_config(body: object) -> dict[str, JSONValue]:
    """JSON-safe build-time config for a graph-IR node.

    Identical to :func:`task_config_of` except for two bodies built around live
    objects: a :class:`SubWorkflow`'s ``inner`` arg (a live ``WorkflowCompiler``
    / ``CompiledWorkflow``, not JSON-serializable) is dropped — the inner
    topology is carried explicitly in ``GraphTaskIR.subworkflow`` — and a
    promoted callable (``_EntryTask``) serializes its ``fn`` as the importable
    ``"module:qualname"`` entrypoint ref (resolved back via importlib when the
    task is reconstructed). A non-importable promoted callable (lambda,
    closure, ``__main__`` function) raises the clear ``ValueError`` from
    :meth:`_EntryTask.entrypoint_ref` rather than a pydantic ValidationError.
    """
    from .promote import _EntryTask
    from .subworkflow import SubWorkflow

    if isinstance(body, _EntryTask):
        return {"fn": body.entrypoint_ref()}
    config = task_config_of(body)
    if isinstance(body, SubWorkflow):
        return {k: v for k, v in config.items() if k != "inner"}
    return config


def _position_of(task: object) -> GraphNodePosition | None:
    """Read a :class:`TaskRegistration`'s optional ``(x, y)`` position."""
    pos = getattr(task, "position", None)
    if pos is None:
        return None
    x, y = pos
    return GraphNodePosition(x=x, y=y)


def _build_edges(spec: Workflow) -> tuple[GraphEdgeIR, ...]:
    """Project the spec's split edge collections into one ``kind``-tagged set.

    ``depends_on`` → ``data``; ``_control_edges`` → ``control``;
    ``_branch_edges`` → ``branch`` (carrying the route label as ``condition``);
    each ``wf.loop`` → ``loop`` edges (back-edge + exit); each ``wf.parallel``
    → ``parallel`` edges (fan-out + join).
    """
    edges: list[GraphEdgeIR] = []
    for t in spec._tasks:
        for dep in t.depends_on:
            edges.append(GraphEdgeIR(source=dep, target=t.name, kind="data"))
    for src, tgt in spec._control_edges:
        edges.append(GraphEdgeIR(source=src, target=tgt, kind="control"))
    for src, label, tgt in spec._branch_edges:
        edges.append(GraphEdgeIR(source=src, target=tgt, kind="branch", condition=label))
    for loop in spec._loops:
        if loop.body:
            edges.append(GraphEdgeIR(source=loop.until, target=loop.body[0], kind="loop"))
        edges.append(GraphEdgeIR(source=loop.until, target=loop.on_exit, kind="loop"))
    for p in spec._parallels:
        edges.append(GraphEdgeIR(source=p.map_over, target=p.body, kind="parallel"))
        edges.append(GraphEdgeIR(source=p.body, target=p.join, kind="parallel"))
    return tuple(edges)


# ``GraphTaskIR.subworkflow`` forward-references ``WorkflowGraphIR`` (defined
# above it), and ``WorkflowGraphIR.tasks`` references ``GraphTaskIR`` — a mutually
# recursive pair. Rebuild both so pydantic resolves the string annotations.
GraphTaskIR.model_rebuild()
WorkflowGraphIR.model_rebuild()
