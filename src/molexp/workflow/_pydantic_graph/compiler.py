"""WorkflowGraphCompiler — single-path CFG compiler (spec 03 §8).

Pipeline:

1. **Data-DAG validation** — ``depends_on`` graph must be acyclic; cycles
   raise :class:`CycleError` with a "data graph" message hinting at control
   edges.
2. **Edge-set construction** — explicit ``wf.control`` / ``wf.branch``
   declarations are bucketed by source; tasks with no explicit control
   declarations get a default :class:`UnconditionalEdges` synthesised from
   the reverse data edges (§2). Mixing branch + unconditional on the same
   source raises :class:`EdgeShapeError`.
3. **Entry resolution** — explicit ``wf.entry(...)`` wins; otherwise
   workflows with any explicit control edge raise
   :class:`EntryAmbiguousError` (§4); pure-data DAGs use the data-zero
   tasks as the entry frontier.
4. **Reachability check** — every registered task must be reachable from
   the entry frontier through control + reverse-data edges; orphans raise
   :class:`UnreachableTaskError` (§8 step 4).
5. **Codegen + Graph construction** — produce one :class:`pydantic_graph.BaseNode`
   subclass per task via :func:`make_task_node_class`, instantiate them, and
   build a ``Graph(nodes=[*all_task_node_classes])`` for IR / snapshot use
   (the runtime drives the frontier itself; ``Graph.run`` is **not** invoked).
"""

from __future__ import annotations

import graphlib
from collections import defaultdict
from typing import Any

from pydantic_graph import BaseNode, Graph

from ..spec import ParallelDecl, TaskRegistration, WorkflowSpec
from ..types import (
    BranchEdges,
    CycleError,
    EdgeShapeError,
    EntryAmbiguousError,
    OutEdges,
    UnconditionalEdges,
    UnknownTaskError,
    UnreachableTaskError,
)
from .node import END_TARGET, WorkflowStep, make_task_node_class
from .state import WorkflowDeps, WorkflowState


class CompiledWorkflow:
    """Output of :meth:`WorkflowGraphCompiler.compile`.

    Carries the per-task BaseNode classes/instances, the compiled
    out-edge map, the resolved entry frontier, and a Graph object
    used solely for IR / snapshot dumps. Callers obtain a fresh
    :class:`WorkflowDeps` per execution via :meth:`make_deps`.
    """

    def __init__(
        self,
        *,
        graph: Graph[WorkflowState, WorkflowDeps, WorkflowState],
        node_classes: list[type[BaseNode[WorkflowState, WorkflowDeps, WorkflowState]]],
        task_by_name: dict[str, BaseNode[WorkflowState, WorkflowDeps, WorkflowState]],
        registration_by_name: dict[str, TaskRegistration],
        out_edges: dict[str, OutEdges],
        entry_frontier: tuple[str, ...],
        loop_max_iters: dict[str, int],
        parallel_decls: dict[str, ParallelDecl],
    ) -> None:
        self.graph = graph
        self.node_classes = node_classes
        self.task_by_name = task_by_name
        self.registration_by_name = registration_by_name
        self.out_edges = out_edges
        self.entry_frontier = entry_frontier
        self.loop_max_iters = loop_max_iters
        self.parallel_decls = parallel_decls

    def make_deps(
        self,
        run: Any = None,
        run_context: Any = None,
        config: Any = None,
        user_deps: Any = None,
        remote_executor: Any = None,
        run_dir: Any = None,
    ) -> WorkflowDeps:
        return WorkflowDeps(
            run=run,
            run_context=run_context,
            config=config,
            user_deps=user_deps,
            remote_executor=remote_executor,
            run_dir=run_dir,
            task_by_name=dict(self.task_by_name),
            out_edges=dict(self.out_edges),
            entry_frontier=self.entry_frontier,
            loop_max_iters=dict(self.loop_max_iters),
            parallel_decls=dict(self.parallel_decls),
        )


class WorkflowGraphCompiler:
    """Compile a :class:`WorkflowSpec` into a :class:`CompiledWorkflow`."""

    def compile(self, spec: WorkflowSpec) -> CompiledWorkflow:
        self._validate_data_dag(spec)
        out_edges = self._compile_edge_sets(spec)
        entry_frontier = self._resolve_entry_frontier(spec, out_edges)
        self._check_reachability(spec, out_edges, entry_frontier)

        node_classes: list[type[BaseNode[WorkflowState, WorkflowDeps, WorkflowState]]] = []
        task_by_name: dict[str, BaseNode[WorkflowState, WorkflowDeps, WorkflowState]] = {}
        registration_by_name: dict[str, TaskRegistration] = {}
        for reg in spec._tasks:
            edge_set = out_edges[reg.name]
            cls = make_task_node_class(name=reg.name, registration=reg, edge_set=edge_set)
            node_classes.append(cls)
            task_by_name[reg.name] = _instantiate_task_node(reg, cls)
            registration_by_name[reg.name] = reg

        # Only ``WorkflowStep`` needs registration in the pydantic-graph
        # Graph: it is the per-frame frontier scheduler that pydantic-graph
        # drives (``WorkflowStep(0) → WorkflowStep(1) → … → End``). The
        # per-task BaseNode subclasses are invoked via ``await node.run(ctx)``
        # from inside ``WorkflowStep`` and never appear in pydantic-graph's
        # node chain, so they are *not* registered here — registration would
        # trigger pydantic-graph's strict ``run`` return-type check, which
        # cannot accept the loose ``Any`` shape the user-facing ``Task.run``
        # uses by design (see ``workflow/task.py``).
        graph_nodes: list[type[BaseNode[WorkflowState, WorkflowDeps, WorkflowState]]] = (
            [WorkflowStep] if node_classes else [_PlaceholderEnd]
        )
        graph: Graph[WorkflowState, WorkflowDeps, WorkflowState] = Graph(
            nodes=graph_nodes,
            state_type=WorkflowState,
            run_end_type=WorkflowState,
        )

        loop_max_iters = {loop.until: loop.max_iters for loop in spec._loops}
        parallel_decls = {par.body: par for par in spec._parallels}

        compiled = CompiledWorkflow(
            graph=graph,
            node_classes=node_classes,
            task_by_name=task_by_name,
            registration_by_name=registration_by_name,
            out_edges=out_edges,
            entry_frontier=entry_frontier,
            loop_max_iters=loop_max_iters,
            parallel_decls=parallel_decls,
        )
        # Surface the compiled node classes back on the spec so tests and
        # tooling can introspect (``test_no_trampoline_node``).
        spec._compiled_node_classes = list(node_classes)  # type: ignore[attr-defined]
        return compiled

    # ── Stage 1 ─ data DAG ──────────────────────────────────────────────

    def _validate_data_dag(self, spec: WorkflowSpec) -> list[str]:
        names = {t.name for t in spec._tasks}
        for t in spec._tasks:
            for dep in t.depends_on:
                if dep not in names:
                    raise UnknownTaskError(
                        f"Task {t.name!r} depends_on unknown task {dep!r}; "
                        f"registered: {sorted(names)}"
                    )
        graph: dict[str, set[str]] = {t.name: set(t.depends_on) for t in spec._tasks}
        try:
            return list(graphlib.TopologicalSorter(graph).static_order())
        except graphlib.CycleError as exc:
            raise CycleError(
                f"Workflow {spec.name!r}: data graph contains a cycle ({exc}); "
                f"`depends_on` must be a DAG. Express loops with a control edge "
                f"(wf.control / wf.branch) instead."
            ) from exc

    # ── Stage 2 ─ edge sets ─────────────────────────────────────────────

    def _compile_edge_sets(self, spec: WorkflowSpec) -> dict[str, OutEdges]:
        names = {t.name for t in spec._tasks}
        ctrl_by_src: dict[str, list[str]] = defaultdict(list)
        branch_by_src: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for src, tgt in spec._control_edges:
            self._verify_known(src, names, "control edge source")
            if tgt != END_TARGET:
                self._verify_known(tgt, names, "control edge target")
            ctrl_by_src[src].append(tgt)
        for src, label, tgt in spec._branch_edges:
            self._verify_known(src, names, "branch edge source")
            if tgt != END_TARGET:
                self._verify_known(tgt, names, "branch edge target")
            branch_by_src[src].append((label, tgt))

        # Spec 05 §4 — expand each parallel decl into two unconditional
        # control edges (``map_over → body`` and ``body → join``), and
        # cross-validate against loop / branch / control / depends_on
        # collisions on the body task.
        loop_until_names = {loop.until for loop in spec._loops}
        loop_body_names: set[str] = set()
        for loop in spec._loops:
            loop_body_names.update(loop.body)
        parallel_body_names: set[str] = set()
        for par in spec._parallels:
            self._verify_known(par.map_over, names, "parallel map_over")
            self._verify_known(par.body, names, "parallel body")
            self._verify_known(par.join, names, "parallel join")
            if par.body in parallel_body_names:
                raise EdgeShapeError(
                    f"Task {par.body!r} is the body of two wf.parallel decls. "
                    "Each body task may participate in at most one parallel."
                )
            parallel_body_names.add(par.body)
            if par.body in loop_until_names or par.body in loop_body_names:
                raise EdgeShapeError(
                    f"Task {par.body!r} is the body of a wf.parallel AND part of a "
                    "wf.loop (either body or until). Nesting is rejected; "
                    "compose by writing a wrapper task instead."
                )
            if par.body in ctrl_by_src or par.body in branch_by_src:
                raise EdgeShapeError(
                    f"Task {par.body!r} is the body of a wf.parallel AND has explicit "
                    "wf.control / wf.branch declarations. The parallel primitive owns "
                    "the body's outgoing edges; remove the manual declarations."
                )
            body_reg = next((t for t in spec._tasks if t.name == par.body), None)
            if body_reg is not None and body_reg.depends_on:
                raise EdgeShapeError(
                    f"Task {par.body!r} is the body of a wf.parallel AND declares "
                    f"depends_on={body_reg.depends_on!r}. The parallel primitive owns "
                    "the body's data wiring (each invocation receives one element of "
                    "map_over via ctx.inputs); remove the explicit depends_on."
                )
            ctrl_by_src[par.map_over].append(par.body)
            ctrl_by_src[par.body].append(par.join)
            # Inject ``body`` as a data dep of ``join`` so the existing
            # _collect_upstream_outputs path threads the aggregated list
            # into ``ctx.inputs``. Idempotent.
            join_reg = next((t for t in spec._tasks if t.name == par.join), None)
            if join_reg is not None and par.body not in join_reg.depends_on:
                join_reg.depends_on.append(par.body)

        # Spec 04 §4 — expand each loop into branch edges on `until`.
        # The body chain (body[i] → body[i+1] → … → body[-1] → until) is
        # the user's responsibility to wire via depends_on or
        # ``wf.control``. The loop only owns the back-edge branch on
        # ``until``: ``{"continue": body[0], "exit": on_exit}``.
        for loop in spec._loops:
            for body_name in loop.body:
                self._verify_known(body_name, names, "loop body task")
            self._verify_known(loop.until, names, "loop until task")
            if loop.on_exit != END_TARGET:
                self._verify_known(loop.on_exit, names, "loop on_exit task")
            if loop.until in branch_by_src:
                raise EdgeShapeError(
                    f"Task {loop.until!r} is the `until` of a wf.loop AND has "
                    "explicit branch edges (wf.branch). Pick one form: let the "
                    "loop own the branch, or remove the loop and wire branches "
                    "manually."
                )
            branch_by_src[loop.until].append(("continue", loop.body[0]))
            branch_by_src[loop.until].append(("exit", loop.on_exit))

        out_edges: dict[str, OutEdges] = {}
        for t in spec._tasks:
            has_ctrl = t.name in ctrl_by_src
            has_br = t.name in branch_by_src
            if has_ctrl and has_br:
                raise EdgeShapeError(
                    f"Task {t.name!r} mixes unconditional and branch out-edges. "
                    "Pick one form: either wf.control(...) repeatedly OR "
                    "wf.branch(...) with routes={...} (spec 03 §3)."
                )
            if has_br:
                routes: dict[str, str] = {}
                for lbl, tgt in branch_by_src[t.name]:
                    if lbl in routes:
                        raise EdgeShapeError(f"Task {t.name!r}: duplicate branch label {lbl!r}")
                    routes[lbl] = tgt
                out_edges[t.name] = BranchEdges(routes=routes)
            elif has_ctrl:
                out_edges[t.name] = UnconditionalEdges(targets=tuple(ctrl_by_src[t.name]))
            else:
                # Default: synthesise unconditional edges from reverse data edges (§2).
                downstream = tuple(
                    other.name for other in spec._tasks if t.name in other.depends_on
                )
                out_edges[t.name] = UnconditionalEdges(targets=downstream)
        return out_edges

    @staticmethod
    def _verify_known(name: str, registered: set[str], role: str) -> None:
        if name not in registered:
            raise UnknownTaskError(
                f"{role} references unregistered task {name!r}; registered: {sorted(registered)}"
            )

    # ── Stage 3 ─ entry frontier ────────────────────────────────────────

    def _resolve_entry_frontier(
        self,
        spec: WorkflowSpec,
        out_edges: dict[str, OutEdges],
    ) -> tuple[str, ...]:
        names = {t.name for t in spec._tasks}

        if spec._entries:
            for e in spec._entries:
                if e not in names:
                    raise UnknownTaskError(
                        f"wf.entry({e!r}) references an unregistered task; "
                        f"registered: {sorted(names)}"
                    )
            return tuple(spec._entries)

        has_explicit_control = bool(spec._control_edges) or bool(spec._branch_edges)
        if has_explicit_control:
            ctrl_targets = {tgt for _, tgt in spec._control_edges}
            branch_targets = {tgt for _, _, tgt in spec._branch_edges}
            incoming_control = ctrl_targets | branch_targets
            candidates = sorted(
                t.name for t in spec._tasks if not t.depends_on and t.name not in incoming_control
            )
            if candidates:
                hint = f"Candidates (no incoming edges): {candidates}"
            else:
                hint = "No candidate entries (every task has an incoming edge)."
            raise EntryAmbiguousError(
                f"Workflow {spec.name!r} has explicit control edges but no "
                f"wf.entry(...) declaration. Declare it explicitly. {hint}"
            )

        return tuple(t.name for t in spec._tasks if not t.depends_on)

    # ── Stage 4 ─ reachability ──────────────────────────────────────────

    def _check_reachability(
        self,
        spec: WorkflowSpec,
        out_edges: dict[str, OutEdges],
        entry_frontier: tuple[str, ...],
    ) -> None:
        if not spec._tasks:
            return

        reverse_data: dict[str, list[str]] = defaultdict(list)
        for t in spec._tasks:
            for dep in t.depends_on:
                reverse_data[dep].append(t.name)

        reachable: set[str] = set()
        stack: list[str] = list(entry_frontier)
        while stack:
            n = stack.pop()
            if n in reachable or n == END_TARGET:
                continue
            reachable.add(n)
            edge_set = out_edges.get(n)
            if isinstance(edge_set, UnconditionalEdges):
                stack.extend(t for t in edge_set.targets if t != END_TARGET)
            elif isinstance(edge_set, BranchEdges):
                stack.extend(t for t in edge_set.routes.values() if t != END_TARGET)
            stack.extend(reverse_data.get(n, []))

        registered = {t.name for t in spec._tasks}
        unreachable = sorted(registered - reachable)
        if unreachable:
            raise UnreachableTaskError(
                f"Tasks unreachable from workflow entry through control / data edges: "
                f"{unreachable}. Entry frontier: {list(entry_frontier)}. "
                "Either remove the orphans or add a control edge reaching them."
            )


def _instantiate_task_node(
    reg: TaskRegistration,
    cls: type[BaseNode[WorkflowState, WorkflowDeps, WorkflowState]],
) -> BaseNode[WorkflowState, WorkflowDeps, WorkflowState]:
    """Pick the right runtime instance for a task.

    Decorator-style (``@wf.task``) uses a synthesised
    :class:`_CallableTask` / :class:`_StreamableTask` subclass with no
    required init args, so a fresh ``cls()`` is fine.

    OOP-style (user's own ``class FetchTask(Task)`` with a custom
    ``__init__``) cannot be re-instantiated without their args. We reuse
    the original instance and stamp the per-registration metadata onto
    it so :func:`_task_run` finds it via ``getattr(self, ...)``.
    """
    from ..task import Task as _UserTask

    if isinstance(reg.fn_or_class, _UserTask):
        instance = reg.fn_or_class
        # The per-registration subclass make_task_node_class returned has
        # ``_molexp_*`` stamped via ``type(name, bases, namespace)``. Type
        # checkers don't see dynamic-namespace attributes, so read them
        # via ``getattr`` (and write to the instance via ``setattr``) —
        # this keeps ty silent without sprinkling per-line ignores.
        for attr in ("_molexp_task_name", "_molexp_registration", "_molexp_edge_set"):
            setattr(instance, attr, getattr(cls, attr))
        return instance
    return cls()


class _PlaceholderEnd(BaseNode[WorkflowState, WorkflowDeps, WorkflowState]):
    """Inserted only when the workflow registers zero tasks — keeps Graph happy."""

    async def run(self, ctx: Any) -> Any:
        from pydantic_graph import End

        return End(ctx.state)
