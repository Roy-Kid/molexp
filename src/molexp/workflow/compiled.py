"""CompiledWorkflow — the single rich artifact emitted by ``WorkflowCompiler.compile``.

This dissolves the old ``Workflow`` god-object. It carries everything the
compiler derives in one pass:

- the topology (tasks + control/branch/loop/parallel/entry decls),
- the executable ``graph`` (a layer-private ``ExecutionPlan`` lowered by the
  engine compiler; only the workflow runtime reads it),
- per-task ``snapshots`` (one :class:`TaskSnapshot` each),
- the ``version`` (:class:`WorkflowVersion`, reusing the snapshot code-hash),
- the experiment ``binding`` (``WorkflowBinding | None``).

It is a **plain class**, not a ``pydantic.BaseModel``: it holds live task
callables, which makes this a runtime container by definition (per the
CLAUDE.md "Pydantic vs plain class" rule). It is immutable by discipline —
construct it via :meth:`WorkflowCompiler.compile` and do not mutate it.

Execution (``execute`` / ``start`` / ``run_on``) lives on the runtime, not
here; binding lives in :class:`WorkflowBindingRegistry`, not on a global.
``graph`` is layer-private — only the workflow runtime reads it; layers above
``workflow`` use only the codec / introspection methods below.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import cached_property
from typing import TYPE_CHECKING, Protocol

from ._graph_decl import (
    LoopDecl,
    ParallelDecl,
    TaskRegistration,
    _BoundaryStubTask,
)
from .protocols import JSONValue, TaskOutput

if TYPE_CHECKING:
    from ._pydantic_graph.compiler import CompiledGraph
    from .binding import WorkflowBinding
    from .ir import WorkflowGraphIR
    from .registry import TaskTypeRegistry
    from .snapshot import TaskSnapshot
    from .version import WorkflowVersion


class _ExperimentLike(Protocol):
    """Duck-typed handle to anything with a stable string ``id``."""

    @property
    def id(self) -> str: ...


class CompiledWorkflow:
    """Frozen-by-discipline compiled workflow artifact (see module docstring)."""

    def __init__(
        self,
        *,
        name: str,
        workflow_id: str,
        version_label: str,
        tasks: list[TaskRegistration],
        graph: CompiledGraph,
        snapshots: Mapping[str, TaskSnapshot],
        version: WorkflowVersion,
        mode: str = "batch",
        entries: tuple[str, ...] = (),
        control_edges: tuple[tuple[str, str], ...] = (),
        branch_edges: tuple[tuple[str, str, str], ...] = (),
        loops: tuple[LoopDecl, ...] = (),
        parallels: tuple[ParallelDecl, ...] = (),
        reducer: tuple[str, Callable[..., TaskOutput]] | None = None,
        binding: WorkflowBinding | None = None,
    ) -> None:
        self.name = name
        self.workflow_id = workflow_id
        self.version_label = version_label
        self._mode = mode
        # ``graph`` is the layer-private ExecutionPlan; only the runtime reads it.
        self.graph = graph
        self.snapshots = snapshots
        self.version = version
        self.binding = binding
        # Topology data (subsumes the old Workflow spec) — read by the codec,
        # the full-graph IR exporter, and subgraph extraction.
        self._tasks = tasks
        self._entries = entries
        self._control_edges = control_edges
        self._branch_edges = branch_edges
        self._loops = loops
        self._parallels = parallels
        self._reducer = reducer

    # ── Derived topology maps (built once; the artifact is frozen) ─────────

    @cached_property
    def registration_by_name(self) -> Mapping[str, TaskRegistration]:
        """``task_name → TaskRegistration`` — derived once and reused across
        every execution (the runtime's per-run deps read it)."""
        return {t.name: t for t in self._tasks}

    @cached_property
    def parallel_decls_by_body(self) -> Mapping[str, ParallelDecl]:
        """``body_task_name → ParallelDecl`` — derived once, reused per run."""
        return {par.body: par for par in self._parallels}

    @cached_property
    def loop_max_iters(self) -> Mapping[str, int]:
        """``until_task_name → max_iters`` — derived once, reused per run."""
        return {loop.until: loop.max_iters for loop in self._loops}

    # ── Boundary introspection ────────────────────────────────────────────

    @property
    def boundary_names(self) -> frozenset[str]:
        """Names of boundary-stub tasks (empty for full pipelines)."""
        return frozenset(
            t.name for t in self._tasks if isinstance(t.fn_or_class, _BoundaryStubTask)
        )

    @property
    def non_boundary_names(self) -> frozenset[str]:
        """Names of runnable (non-stub) tasks — complement of :attr:`boundary_names`."""
        return frozenset(t.name for t in self._tasks) - self.boundary_names

    # ── Cross-replicate reducer ───────────────────────────────────────────

    @property
    def reducer_dimension(self) -> str | None:
        """Dimension declared on ``@wf.reduce(over=...)``; None if no reducer."""
        return self._reducer[0] if self._reducer is not None else None

    def run_reducer(self, replicate_outputs: list[TaskOutput]) -> TaskOutput:
        """Invoke the registered ``@wf.reduce`` on a list of replicate outputs."""
        if self._reducer is None:
            raise LookupError(
                f"Workflow {self.name!r}: no reducer registered (call @wf.reduce(over=...))"
            )
        return self._reducer[1](replicate_outputs)

    # ── Representation codec (folded from spec 01) ─────────────────────────

    def to_ir(self, *, strict: bool = True) -> dict[str, JSONValue]:
        """Serialize to the JSON IR (data-DAG **wire** format). Delegates to ``default_codec``.

        This is the persistence / round-trip format: it backs :meth:`from_ir`
        and :meth:`to_python`, and under ``strict`` (default ``True``) requires
        a ``task_type`` slug on every task. By design it is the *data DAG only*
        — it omits the parallel fan-out edges (``map_over→body``, ``body→join``)
        and other control/branch/loop topology.

        For a UI canvas, control-flow visualization, or any observability
        consumer that needs the full graph (including those parallel edges),
        use :meth:`to_graph_ir`, which is total and never requires slugs —
        mirroring the :meth:`to_mermaid` / :meth:`to_graph_mermaid` split. Pass
        ``strict=False`` for observability-only serialization that tolerates
        slug-less tasks (``task_type: None``).
        """
        from .codec import default_codec

        return dict(default_codec.spec_to_ir(self, strict=strict))

    def to_python(self) -> str:
        """Render as a runnable Python script (via the IR)."""
        from .codec import default_codec

        return default_codec.spec_to_python(self)

    def to_mermaid(self) -> str:
        """Render the data-DAG as a Mermaid ``flowchart LR`` (via the IR).

        For workflows with control / branch / loop / parallel topology use
        :meth:`to_graph_mermaid`, which renders the full compiled graph.
        """
        from .codec import default_codec

        return default_codec.spec_to_mermaid(self)

    @classmethod
    def from_ir(
        cls,
        data: Mapping[str, JSONValue],
        *,
        registry: TaskTypeRegistry | None = None,
    ) -> CompiledWorkflow:
        """Build a :class:`CompiledWorkflow` from JSON IR. Delegates to ``default_codec``."""
        from .codec import default_codec

        return default_codec.ir_to_spec(data, registry=registry)

    # ── Full-graph IR + diagram export ────────────────────────────────────

    def to_graph_ir(self) -> WorkflowGraphIR:
        """Export the full compiled-graph IR — the blessed entry point for UI / control-flow / observability consumers.

        Emits the complete graph: tasks, data deps, entries, control and branch
        edges, loops, and parallels — including the parallel fan-out edges
        (``map_over→body`` and ``body→join``, tagged ``kind="parallel"``) that
        :meth:`to_ir`'s data-DAG wire format deliberately omits. It is total and
        never requires a ``task_type`` slug.

        Contrast with :meth:`to_ir`, the slug-requiring data-DAG format used for
        persistence / round-trip (:meth:`from_ir`) and script generation
        (:meth:`to_python`). Prefer this method whenever you need the full
        topology rather than the round-trippable wire form — the same split as
        :meth:`to_mermaid` (data DAG) vs :meth:`to_graph_mermaid` (full graph).
        """
        from .ir import build_workflow_graph_ir

        return build_workflow_graph_ir(self)

    def to_graph_mermaid(self, *, direction: str = "LR") -> str:
        """Render the full compiled graph as a Mermaid ``flowchart`` (via :meth:`to_graph_ir`)."""
        from .mermaid import render_workflow_mermaid

        return render_workflow_mermaid(self.to_graph_ir(), direction=direction)

    # ── Subgraph ───────────────────────────────────────────────────────────

    def subgraph(
        self,
        start_nodes: Iterable[str],
        *,
        include_downstream: bool = False,
    ) -> CompiledWorkflow:
        """Return a compiled workflow over a subset of this artifact's nodes.

        For each boundary upstream (a task referenced by a selected node's
        ``depends_on`` but itself outside the selection) a
        ``_BoundaryStubTask`` is registered; supply its value via
        ``runtime.execute(compiled, seed_outputs=...)``.
        """
        selection = list(start_nodes)
        if not selection:
            raise ValueError("CompiledWorkflow.subgraph: start_nodes must not be empty")
        registered = {t.name: t for t in self._tasks}
        unknown = [name for name in selection if name not in registered]
        if unknown:
            raise ValueError(
                f"CompiledWorkflow.subgraph: unknown task name(s) {unknown!r}; "
                f"registered tasks: {sorted(registered)}"
            )

        chosen: set[str] = set(selection)
        if include_downstream:
            forward: dict[str, list[str]] = {name: [] for name in registered}
            for t in self._tasks:
                for dep in t.depends_on:
                    forward.setdefault(dep, []).append(t.name)
            frontier = list(selection)
            while frontier:
                node = frontier.pop()
                for child in forward.get(node, ()):
                    if child not in chosen:
                        chosen.add(child)
                        frontier.append(child)

        new_tasks: list[TaskRegistration] = []
        boundary_upstreams: list[str] = []
        seen_boundaries: set[str] = set()
        for t in self._tasks:
            if t.name not in chosen:
                continue
            new_tasks.append(
                TaskRegistration(
                    name=t.name,
                    fn_or_class=t.fn_or_class,
                    depends_on=list(t.depends_on),
                    is_actor=t.is_actor,
                    remote=t.remote,
                    task_type=t.task_type,
                    dependent_params=t.dependent_params,
                )
            )
            for dep in t.depends_on:
                if dep not in chosen and dep not in seen_boundaries:
                    seen_boundaries.add(dep)
                    boundary_upstreams.append(dep)

        stub_tasks: list[TaskRegistration] = [
            TaskRegistration(name=bname, fn_or_class=_BoundaryStubTask(bname), depends_on=[])
            for bname in boundary_upstreams
        ]
        new_tasks = stub_tasks + new_tasks

        from .compiler import compile_registrations

        return compile_registrations(
            name=self.name,
            version_label=self.version_label,
            tasks=new_tasks,
            mode=self._mode,
        )


__all__ = ["CompiledWorkflow"]
