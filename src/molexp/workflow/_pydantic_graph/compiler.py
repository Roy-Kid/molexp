"""WorkflowGraphCompiler — lower a topology to a genuine pydantic-graph Graph.

Pipeline:

1. **Data-DAG validation** — ``depends_on`` graph must be acyclic; cycles
   raise :class:`CycleError` with a "data graph" message hinting at control
   edges.
2. **Edge-set construction** — explicit ``wf.control`` / ``wf.branch``
   declarations are bucketed by source; tasks with no explicit control
   declarations get a default :class:`UnconditionalEdges` synthesised from
   the reverse data edges. Mixing branch + unconditional on the same
   source raises :class:`EdgeShapeError`.
3. **Entry resolution** — explicit ``wf.entry(...)`` wins; otherwise
   workflows with any explicit control edge raise
   :class:`EntryAmbiguousError`; pure-data DAGs use the data-zero tasks
   as the entry frontier.
4. **Reachability check** — every registered task must be reachable from
   the entry frontier through control + reverse-data edges; orphans raise
   :class:`UnreachableTaskError`.
5. **Lowering** — :meth:`_build_graph` lowers the compiled out-edges,
   entry frontier, loops and parallels into a real
   :class:`pydantic_graph.graph_builder.Graph` with **one Step per task**.
   Control flow rides pg primitives: edges (data/control deps), ``Join``
   (multi-dependency fan-in), map-Fork + ``Join`` (``wf.parallel``), and
   ``Decision`` (``wf.branch`` / ``wf.loop`` routing).

molexp tasks read their inputs from the shared, mutated
:class:`WorkflowState` ``results`` dict (not edge tokens); edges only
express trigger / ordering. The compiler is the **sole** sanctioned
``import pydantic_graph`` site together with the rest of
``workflow/_pydantic_graph/``.
"""

from __future__ import annotations

import graphlib
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

from pydantic_graph.graph_builder import GraphBuilder
from pydantic_graph.join import reduce_dict_update, reduce_null
from pydantic_graph.step import StepContext

from .._graph_decl import ParallelDecl, WorkflowTopology
from ..types import (
    BranchEdges,
    CycleError,
    EdgeShapeError,
    EntryAmbiguousError,
    LoopMaxItersExceeded,
    Next,
    OutEdges,
    ParallelExecutionError,
    UnconditionalEdges,
    UnknownRouteError,
    UnknownTaskError,
    UnreachableTaskError,
)
from .node import (
    END_TARGET,
    NO_OUTPUT,
    TakeEnd,
    TakeLabel,
    _classify_return,
    _EndTok,
    _Failure,
    _Trigger,
    run_task_body,
)
from .state import WorkflowDeps, WorkflowState

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_graph.graph_builder import Graph
    from pydantic_graph.step import StepFunction

    # The compiled workflow graph type. Re-exported here so layer modules
    # that may not import ``pydantic_graph`` directly (e.g. ``compiled.py``,
    # guarded by the encapsulation seam) can name the type via the workflow
    # layer instead of reaching for ``pydantic_graph.graph_builder``.
    CompiledGraph = Graph[WorkflowState, WorkflowDeps, None, None]

# A graph node handle is any value GraphBuilder.add_edge accepts as a
# destination — a Step, Join, Decision, or the start/end node markers.
# pydantic-graph types these as overlapping generics; we keep the local
# alias loose to avoid over-specifying the builder's internal generics.


class WorkflowGraphCompiler:
    """Lower a :class:`WorkflowTopology` into a genuine pydantic-graph ``Graph``.

    Internal helper invoked exactly once by
    :meth:`molexp.workflow.compiler.WorkflowCompiler.compile`. The returned
    ``Graph`` carries one Step per task; the runtime drives it via
    ``graph.run(state=..., deps=..., inputs=None)``.
    """

    def compile(self, spec: WorkflowTopology) -> Graph[WorkflowState, WorkflowDeps, None, None]:
        self._validate_data_dag(spec)
        out_edges = self._compile_edge_sets(spec)
        entry_frontier = self._resolve_entry_frontier(spec)
        self._check_reachability(spec, out_edges, entry_frontier)
        return self._build_graph(spec, out_edges, entry_frontier)

    # ── Stage 5 ─ pydantic-graph lowering ───────────────────────────────

    def _build_graph(
        self,
        spec: WorkflowTopology,
        out_edges: dict[str, OutEdges],
        entry_frontier: tuple[str, ...],
    ) -> Graph[WorkflowState, WorkflowDeps, None, None]:
        """Lower compiled edges/entries into a real ``pydantic_graph`` Graph.

        Lowering rules (see module docstring):

        * Entry tasks → edge from ``start_node`` (multiple → auto fork).
        * ``UnconditionalEdges`` → edge into each target's incoming node;
          ``_end`` → ``end_node``.
        * Indegree>1 task → a ``reduce_null`` ``Join`` placed before its
          Step; all upstreams edge into the Join.
        * Terminal task (no out-edges) → edge to ``end_node``.
        * ``BranchEdges`` task → Step returns a ``Next`` token; a
          ``Decision`` routes it per label.
        * ``wf.parallel`` → ``map_over`` Step → mapping-edge fan-out into a
          body-wrapper Step → ``reduce_dict_update`` ``Join`` → collector
          Step → user join Step.
        """
        gb = GraphBuilder[WorkflowState, WorkflowDeps, None, None](
            name=spec.name or "workflow",
            state_type=WorkflowState,
            deps_type=WorkflowDeps,
            output_type=type(None),
        )

        parallel_decls = {par.body: par for par in spec._parallels}
        # ``map_over → ParallelDecl`` so the map_over Step uses the dedicated
        # mapping edge instead of a routing Decision.
        parallel_by_map_over = {par.map_over: par for par in spec._parallels}

        # ── Back-edge detection (cycles: wf.loop / self-loop) ───────────
        # A back-edge re-enters an already-visited node, forming a cycle
        # (sequential re-run, not a fan-in). It must route directly to the
        # target Step — never through a coalescing Join — and is excluded
        # from indegree counting so it does not synthesise a spurious Join.
        back_edges = self._compute_back_edges(out_edges, entry_frontier, parallel_decls)

        # ── Indegree coalescing: count incoming edges per task ──────────
        indegree = self._compute_indegree(
            spec, out_edges, entry_frontier, parallel_decls, back_edges
        )

        # ── Build Step nodes (skip parallel body tasks — they get a
        #    dedicated wrapper Step instead of the plain body Step) ──────
        steps: dict[str, object] = {}
        for t in spec._tasks:
            if t.name in parallel_decls:
                continue
            steps[t.name] = gb.step(
                self._make_step_fn(t.name, out_edges[t.name]),
                node_id=t.name,
            )

        # ── Join nodes for indegree>1 tasks (the task's "incoming node") ─
        joins: dict[str, object] = {}
        for t in spec._tasks:
            if t.name in parallel_decls:
                continue
            if indegree.get(t.name, 0) > 1:
                joins[t.name] = gb.join(
                    reduce_null,
                    initial_factory=lambda: None,
                    node_id=f"{t.name}__join",
                )

        def incoming_of(name: str) -> object:
            """The node an edge into *name* should target (Join if coalesced)."""
            if name == END_TARGET:
                return gb.end_node
            if name in joins:
                return joins[name]
            return steps[name]

        def route_to(src: str, tgt: str) -> object:
            """Destination for the edge ``src → tgt``.

            Back-edges (cycles) bypass any coalescing Join and re-enter the
            target Step directly; forward edges use :func:`incoming_of`.
            """
            if tgt != END_TARGET and (src, tgt) in back_edges:
                return steps[tgt]
            return incoming_of(tgt)

        # ── Wire entry edges from start_node ─────────────────────────────
        # A ``wf.parallel`` body is reached only via its map_over fan-out;
        # it is never a graph entry even when it has no ``depends_on``.
        entries = [e for e in entry_frontier if e not in parallel_decls]
        if not entries:
            # Empty workflow (no tasks / no entries) — the graph must still
            # reach end_node for the builder to validate.
            gb.add_edge(gb.start_node, gb.end_node)
        for entry in entries:
            gb.add_edge(gb.start_node, incoming_of(entry))

        # ── Wire Join → Step edges (the coalesced fan-in) ────────────────
        for name, join in joins.items():
            gb.add_edge(join, steps[name])

        # ── Per-Step routing via Decision ────────────────────────────────
        # Every non-parallel, non-map_over Step is routed through a
        # ``Decision``: branch tasks match a ``Next(label)`` per route;
        # unconditional tasks match ``_Trigger`` (fan out to all targets) or
        # ``_EndTok`` (terminate). map_over Steps keep the dedicated parallel
        # mapping edge (wired below) instead of a Decision.
        for t in spec._tasks:
            if t.name in parallel_decls or t.name in parallel_by_map_over:
                continue
            edge_set = out_edges[t.name]
            if isinstance(edge_set, BranchEdges):
                self._wire_branch(gb, t.name, steps[t.name], edge_set, route_to)
            else:
                assert isinstance(edge_set, UnconditionalEdges)
                self._wire_unconditional(
                    gb, t.name, steps[t.name], edge_set, route_to, parallel_decls
                )

        # ── Parallel subgraphs ───────────────────────────────────────────
        for par in spec._parallels:
            self._wire_parallel(gb, par, steps, incoming_of)

        return gb.build()

    # ── Step factory ─────────────────────────────────────────────────────

    @staticmethod
    def _make_step_fn(
        name: str,
        edge_set: OutEdges,
    ) -> StepFunction[WorkflowState, WorkflowDeps, object, object]:
        """Build the ``async`` Step body for task *name*.

        The Step runs the user body (unless seeded), records the output into
        the shared ``state.results`` in place, and returns the routing token
        consumed by the per-Step ``Decision``:

        * branch task → ``Next(label)`` (one per declared route) or
          ``_EndTok`` (``End()`` / ``_end`` route);
        * unconditional task → ``_Trigger`` (advance to all targets) or
          ``_EndTok`` (the body yielded ``End()``).
        """
        branch_edges = edge_set if isinstance(edge_set, BranchEdges) else None

        async def _step(ctx: StepContext[WorkflowState, WorkflowDeps, object]) -> object:
            state = ctx.state
            deps = ctx.deps

            if name in state.seeded:
                # Body already produced its value (seed_outputs); skip
                # running it but still route as if it returned that value.
                raw: object = state.results.get(name)
            else:
                raw = await run_task_body(name, deps, state)

            recorded_value, dispatch = _classify_return(raw, edge_set, task_name=name)
            if recorded_value is not NO_OUTPUT:
                state.record(name, recorded_value)
            else:
                state.completed.add(name)

            # wf.loop max_iters guard — increment the until-task's counter
            # whenever it would route "continue"; once at the cap, emit
            # LoopMaxItersExceeded and force "exit".
            if isinstance(dispatch, TakeLabel) and dispatch.label == "continue":
                max_iters = deps.loop_max_iters.get(name)
                if max_iters is not None:
                    new_count = state.loop_counters.get(name, 0) + 1
                    state.loop_counters[name] = new_count
                    if new_count >= max_iters:
                        warnings.warn(
                            LoopMaxItersExceeded(
                                f"Loop guarded by {name!r} reached "
                                f"max_iters={max_iters}; forcing Next('exit'). "
                                "Increase max_iters if more iterations are needed."
                            ),
                            stacklevel=2,
                        )
                        dispatch = TakeLabel("exit")

            if isinstance(dispatch, TakeEnd):
                return _EndTok()

            if branch_edges is not None:
                if isinstance(dispatch, TakeLabel):
                    routes = branch_edges.routes
                    if dispatch.label not in routes:
                        raise UnknownRouteError(
                            f"Task {name!r} returned Next({dispatch.label!r}) but "
                            f"its declared routes are: {sorted(routes)}."
                        )
                    if routes[dispatch.label] == END_TARGET:
                        return _EndTok()
                    return Next(dispatch.label)
                # _classify_return already raises MissingRouteError for a
                # branch task returning a plain value; this arm is defensive.
                raise UnknownRouteError(
                    f"Task {name!r} has branch out-edges but produced no route."
                )

            # Unconditional task: token is just a trigger to its targets.
            return _Trigger()

        return _step

    # ── Branch wiring ──────────────────────────────────────────────────────

    @staticmethod
    def _wire_branch(
        gb: GraphBuilder[WorkflowState, WorkflowDeps, None, None],
        src: str,
        step: object,
        edge_set: BranchEdges,
        route_to: Callable[[str, str], object],
    ) -> None:
        decision = gb.decision(node_id=f"{src}__decision")
        for label, target in edge_set.routes.items():
            if target == END_TARGET:
                continue  # End routes arrive as _EndTok, handled below
            decision = decision.branch(
                gb.match(
                    Next,
                    matches=lambda v, lbl=label: isinstance(v, Next) and v.label == lbl,
                ).to(route_to(src, target))
            )
        # ``End()`` / ``_end``-routed branches arrive as ``_EndTok``.
        decision = decision.branch(gb.match(_EndTok).to(gb.end_node))
        gb.add_edge(step, decision)

    # ── Unconditional wiring ────────────────────────────────────────────────

    @staticmethod
    def _wire_unconditional(
        gb: GraphBuilder[WorkflowState, WorkflowDeps, None, None],
        src: str,
        step: object,
        edge_set: UnconditionalEdges,
        route_to: Callable[[str, str], object],
        parallel_decls: dict[str, ParallelDecl],
    ) -> None:
        targets = [
            tgt for tgt in edge_set.targets if tgt != END_TARGET and tgt not in parallel_decls
        ]
        decision = gb.decision(node_id=f"{src}__decision")
        # ``End()`` from the body terminates regardless of declared targets.
        decision = decision.branch(gb.match(_EndTok).to(gb.end_node))
        if targets:
            dests = [route_to(src, tgt) for tgt in targets]
            decision = decision.branch(gb.match(_Trigger).to(*dests))
        else:
            # Terminal task — a trigger also routes to end so the graph
            # reaches end_node (builder requires every node to reach end).
            decision = decision.branch(gb.match(_Trigger).to(gb.end_node))
        gb.add_edge(step, decision)

    # ── Parallel wiring ────────────────────────────────────────────────────

    def _wire_parallel(
        self,
        gb: GraphBuilder[WorkflowState, WorkflowDeps, None, None],
        par: ParallelDecl,
        steps: dict[str, object],
        incoming_of: Callable[[str], object],
    ) -> None:
        body = par.body
        collect = gb.join(
            reduce_dict_update,
            initial_factory=dict,
            node_id=f"{body}__collect",
        )
        body_wrapper = gb.step(self._make_parallel_body_fn(body), node_id=body)
        collector = gb.step(self._make_parallel_collector_fn(body), node_id=f"{body}__collector")

        # map_over → (enumerate) → map fan-out → body_wrapper
        gb.add(
            gb.edge_from(steps[par.map_over])
            .transform(lambda ctx: list(enumerate(ctx.state.results.get(par.map_over) or [])))
            .map(downstream_join_id=collect.id)
            .to(body_wrapper)
        )
        gb.add_edge(body_wrapper, collect)
        gb.add_edge(collect, collector)
        gb.add_edge(collector, incoming_of(par.join))

    @staticmethod
    def _make_parallel_body_fn(
        body: str,
    ) -> StepFunction[WorkflowState, WorkflowDeps, tuple[int, object], dict[int, object]]:
        """Body wrapper for one ``wf.parallel`` element.

        Receives ``(idx, elem)`` as ``ctx.inputs``, runs the body under the
        per-body :class:`anyio.CapacityLimiter`, and returns ``{idx: out}``
        (or ``{idx: _Failure(exc)}`` on failure — capture-don't-cancel). The
        single-key dict lets ``reduce_dict_update`` merge results in the
        collect ``Join`` while preserving the index.
        """

        async def _body(
            ctx: StepContext[WorkflowState, WorkflowDeps, tuple[int, object]],
        ) -> dict[int, object]:
            idx, elem = ctx.inputs
            limiter = ctx.deps.parallel_limiters.get(body)
            try:
                if limiter is not None:
                    async with limiter:
                        out = await run_task_body(body, ctx.deps, ctx.state, element=elem)
                else:
                    out = await run_task_body(body, ctx.deps, ctx.state, element=elem)
            except Exception as exc:  # capture per-element, aggregate in collector
                return {idx: _Failure(exc)}
            return {idx: out}

        return _body

    @staticmethod
    def _make_parallel_collector_fn(
        body: str,
    ) -> StepFunction[WorkflowState, WorkflowDeps, dict[int, object], None]:
        """Collect the merged ``{idx: out}`` dict into an ordered list.

        Raises :class:`ParallelExecutionError` if any element captured a
        failure; otherwise records ``state.results[body]`` as the
        index-ordered list and the fan-out width in ``state.parallel_runs``.
        """

        async def _collect(
            ctx: StepContext[WorkflowState, WorkflowDeps, dict[int, object]],
        ) -> None:
            merged: dict[int, object] = ctx.inputs or {}
            failures = {i: f.exc for i, f in merged.items() if isinstance(f, _Failure)}
            if failures:
                raise ParallelExecutionError(body=body, failures=failures)
            ordered = [merged[i] for i in sorted(merged)]
            ctx.state.results[body] = ordered
            ctx.state.completed.add(body)
            ctx.state.parallel_runs[body] = len(ordered)

        return _collect

    # ── Back-edges + indegree ────────────────────────────────────────────

    @staticmethod
    def _iter_targets(edge_set: OutEdges) -> list[str]:
        """All edge targets of *edge_set*, excluding ``_end``."""
        if isinstance(edge_set, UnconditionalEdges):
            return [t for t in edge_set.targets if t != END_TARGET]
        if isinstance(edge_set, BranchEdges):
            return [t for t in edge_set.routes.values() if t != END_TARGET]
        return []

    @classmethod
    def _compute_back_edges(
        cls,
        out_edges: dict[str, OutEdges],
        entry_frontier: tuple[str, ...],
        parallel_decls: dict[str, ParallelDecl],
    ) -> set[tuple[str, str]]:
        """Find cycle-forming edges via DFS over the lowered out-edge graph.

        An edge ``(src, tgt)`` is a back-edge iff ``tgt`` is on the current
        DFS recursion stack when the edge is traversed (the classic
        grey-node test). These edges form loops (``wf.loop`` back-branch,
        self-loops) and must re-enter the target Step directly rather than
        through a coalescing Join. Parallel body wiring is owned by the
        parallel primitive and excluded from this walk.
        """
        back_edges: set[tuple[str, str]] = set()
        visited: set[str] = set()
        on_stack: set[str] = set()

        def dfs(node: str) -> None:
            visited.add(node)
            on_stack.add(node)
            if node not in parallel_decls:
                for tgt in cls._iter_targets(out_edges.get(node, UnconditionalEdges(targets=()))):
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

    @staticmethod
    def _compute_indegree(
        spec: WorkflowTopology,
        out_edges: dict[str, OutEdges],
        entry_frontier: tuple[str, ...],
        parallel_decls: dict[str, ParallelDecl],
        back_edges: set[tuple[str, str]],
    ) -> dict[str, int]:
        """Count *forward* incoming edges per task across the edge structure.

        Entry edges from ``start_node`` count as incoming. Back-edges
        (cycles) are excluded — they re-enter the target Step directly and
        must not synthesise a coalescing Join. ``wf.parallel`` targets the
        join task via the collector Step (one incoming edge); the body /
        map_over / join wiring is owned by the parallel and excluded here.
        End targets never count.
        """
        indegree: dict[str, int] = defaultdict(int)
        for entry in entry_frontier:
            indegree[entry] += 1
        for src, edge_set in out_edges.items():
            if src in parallel_decls:
                continue  # body's out-edges owned by the parallel
            for tgt in WorkflowGraphCompiler._iter_targets(edge_set):
                if tgt in parallel_decls or (src, tgt) in back_edges:
                    continue
                indegree[tgt] += 1
        # The collector → join edge is one incoming edge into each parallel join.
        for par in spec._parallels:
            indegree[par.join] += 1
        return dict(indegree)

    # ── Stage 1 ─ data DAG ──────────────────────────────────────────────

    def _validate_data_dag(self, spec: WorkflowTopology) -> list[str]:
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

    def _compile_edge_sets(self, spec: WorkflowTopology) -> dict[str, OutEdges]:
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
        spec: WorkflowTopology,
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
        spec: WorkflowTopology,
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
