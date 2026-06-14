"""WorkflowGraphCompiler — lower a topology to an executable ExecutionPlan.

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
5. **Lowering** — :meth:`_build_plan` lowers the compiled out-edges, entry
   frontier, back-edges (cycles), per-task trigger sources, and the
   ``wf.parallel`` declarations into a frozen
   :class:`~molexp.workflow._pydantic_graph.plan.ExecutionPlan`. The engine
   (:mod:`.engine`) executes the plan with **values-on-edges** semantics:
   each task's inputs are delivered from its upstreams' outputs, a task
   launches exactly when its dependencies are satisfied, and deadlock is
   detected structurally — no timing constants anywhere.

This module is plain structural lowering; it does not import
``pydantic_graph`` (the workflow layer's remaining pg surface is the ``End``
sentinel re-export, confined to ``workflow/_pydantic_graph/``).
"""

from __future__ import annotations

import graphlib
from collections import defaultdict
from typing import TYPE_CHECKING

from .._graph_decl import WorkflowTopology
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
from ._graph_analysis import compute_back_edges, compute_in_sources, compute_recurrent
from .node import END_TARGET
from .plan import ExecutionPlan

if TYPE_CHECKING:
    # The compiled workflow graph type. Re-exported here so layer modules
    # outside the engine package (e.g. ``compiled.py``) can name the type via
    # the workflow layer instead of reaching into the engine package.
    CompiledGraph = ExecutionPlan


class WorkflowGraphCompiler:
    """Lower a :class:`WorkflowTopology` into an executable :class:`ExecutionPlan`.

    Internal helper invoked exactly once by
    :meth:`molexp.workflow.compiler.WorkflowCompiler.compile`. The returned
    plan carries the validated structural graph; the runtime drives it via
    :func:`.engine.run_plan`.
    """

    def compile(self, spec: WorkflowTopology) -> ExecutionPlan:
        self._validate_data_dag(spec)
        out_edges = self._compile_edge_sets(spec)
        entry_frontier = self._resolve_entry_frontier(spec)
        self._check_reachability(spec, out_edges, entry_frontier)
        return self._build_plan(spec, out_edges, entry_frontier)

    # ── Stage 5 ─ structural lowering ────────────────────────────────────

    def _build_plan(
        self,
        spec: WorkflowTopology,
        out_edges: dict[str, OutEdges],
        entry_frontier: tuple[str, ...],
    ) -> ExecutionPlan:
        """Lower compiled edges/entries into a frozen :class:`ExecutionPlan`.

        Derivations (all structural, computed once at compile time):

        * **back-edges** — cycle-forming edges (``wf.loop`` back-branch,
          self-loops) that re-launch their target directly at run time;
        * **in-sources** — each task's forward trigger sources (``START``
          for entries), driving the engine's control-readiness;
        * **recurrent set** — tasks on a control cycle, whose non-chosen
          branch routes must stay live across iterations;
        * **parallel maps** — ``body → decl`` and ``map_over → decl`` so the
          engine fans out / publishes without re-deriving the topology.
        """
        parallel_decls = {par.body: par for par in spec._parallels}
        parallel_by_map_over = {par.map_over: par for par in spec._parallels}

        back_edges = compute_back_edges(out_edges, entry_frontier, parallel_decls)
        in_sources = compute_in_sources(spec, out_edges, entry_frontier, parallel_decls, back_edges)
        recurrent = compute_recurrent(out_edges, parallel_decls)

        return ExecutionPlan(
            name=spec.name or "workflow",
            task_names=tuple(t.name for t in spec._tasks),
            out_edges=dict(out_edges),
            entry_frontier=entry_frontier,
            back_edges=frozenset(back_edges),
            in_sources=in_sources,
            recurrent=recurrent,
            parallels=tuple(spec._parallels),
            parallel_by_body=parallel_decls,
            parallel_by_map_over=parallel_by_map_over,
        )

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
        ctrl_by_src: defaultdict[str, list[str]] = defaultdict(list)
        branch_by_src: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)

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

        self._expand_parallels(spec, names, ctrl_by_src, branch_by_src)
        self._expand_loops(spec, names, branch_by_src)
        return self._assemble_out_edges(spec, ctrl_by_src, branch_by_src)

    def _expand_parallels(
        self,
        spec: WorkflowTopology,
        names: set[str],
        ctrl_by_src: defaultdict[str, list[str]],
        branch_by_src: defaultdict[str, list[tuple[str, str]]],
    ) -> None:
        """Expand each ``wf.parallel`` into ``map_over → body`` / ``body →
        join`` control edges (spec 05 §4), cross-validating against loop /
        branch / control / ``depends_on`` collisions on the body task, and
        injecting ``body`` as a data dep of ``join`` so the aggregated list
        threads into the join's ``ctx.inputs``. Mutates the edge maps."""
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
            join_reg = next((t for t in spec._tasks if t.name == par.join), None)
            if join_reg is not None and par.body not in join_reg.depends_on:
                join_reg.depends_on.append(par.body)

    def _expand_loops(
        self,
        spec: WorkflowTopology,
        names: set[str],
        branch_by_src: defaultdict[str, list[tuple[str, str]]],
    ) -> None:
        """Expand each ``wf.loop`` into a ``{"continue": body[0], "exit":
        on_exit}`` branch on its ``until`` task (spec 04 §4). The body chain
        is the user's wiring; the loop owns only the back-edge branch.
        Mutates ``branch_by_src``."""
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

    def _assemble_out_edges(
        self,
        spec: WorkflowTopology,
        ctrl_by_src: defaultdict[str, list[str]],
        branch_by_src: defaultdict[str, list[tuple[str, str]]],
    ) -> dict[str, OutEdges]:
        """Bucket each task's collected edges into ``BranchEdges`` /
        ``UnconditionalEdges``; tasks with neither default to a fan-out
        synthesised from their reverse data edges (§2)."""
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
