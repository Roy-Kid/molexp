"""Structural workflow engine — values-on-edges execution of an ExecutionPlan.

molexp owns the data flow and the scheduling decision: each completed task's
recorded output rides its trigger edges to its targets, and a task launches
exactly when

1. **control-ready** — every *live* forward in-edge has fired since the task's
   last launch (an in-edge is live until its source is structurally dead or a
   branch routed away from it), and
2. **data-ready** — every declared ``depends_on`` value is present in
   ``state.results`` / ``state.completed``.

Deadlock detection is **structural and deterministic** — zero timing
constants. A control-ready task whose missing dependency is structurally dead
(no live path can ever produce it) raises :class:`WorkflowDeadlockError`
immediately; if the engine quiesces (nothing running, nothing launchable)
while triggered-but-blocked tasks remain, the same error names the
unsatisfiable dependencies.

Semantics preserved from the prior pg-driven lowering:

* branch tasks route one target per ``Next(label)``; non-chosen routes die and
  their downstream-only tasks become structurally dead;
* ``End()`` is frame-scoped — it kills the task's own out-edges; concurrent
  siblings finish and record normally;
* back-edges (``wf.loop`` / self-loops) re-launch their target directly,
  bypassing forward coalescing; ``loop_max_iters`` forces ``Next("exit")``
  with a :class:`LoopMaxItersExceeded` warning at the cap;
* ``wf.parallel`` fans out the body once per ``map_over`` element under the
  per-body capacity limiter, captures per-element failures without cancelling
  siblings, and publishes the index-ordered list before triggering the join —
  the engine owns the gather, so no partial-finalization handling exists;
* per-node status persistence via :func:`.persistence.mark_task_status` is
  byte-compatible with the previous lowering.

This module MUST NOT import ``pydantic_graph``.
"""

from __future__ import annotations

import asyncio
import contextlib
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..types import (
    BranchEdges,
    LoopMaxItersExceeded,
    ParallelExecutionError,
    UnconditionalEdges,
    UnknownRouteError,
    WorkflowDeadlockError,
)
from .node import (
    END_TARGET,
    NO_OUTPUT,
    TakeEnd,
    TakeLabel,
    _classify_return,
    _Failure,
    run_task_body,
)
from .node_cache import run_task_body_cached
from .persistence import mark_task_status

if TYPE_CHECKING:
    from .._graph_decl import ParallelDecl
    from ..protocols import TaskInput, TaskOutput
    from ..types import OutEdges
    from .plan import ExecutionPlan
    from .state import WorkflowDeps, WorkflowState

from .plan import START


def _snapshot_key_of(deps: WorkflowDeps, name: str) -> str | None:
    """The task's ``TaskSnapshot.key`` (code+config identity), if one exists.

    Persisted next to a completed task's outputs in ``workflow.json`` so
    resume seeding can verify the output was produced by the SAME code —
    see :func:`.persistence.filter_resume_seeds`.
    """
    snap = deps.snapshots.get(name)
    return snap.key if snap is not None else None


def _cache_is_active(deps: WorkflowDeps, name: str) -> bool:
    """Gate the per-task cache hook (spec workflow-refactor-04 §Cache hook).

    Caching is engaged for task *name* only when a cache is present, a
    snapshot exists for the task, and the task is a batch ``Task`` (never an
    ``Actor`` / streaming body — those are never cached).
    """
    if deps.cache is None or name not in deps.snapshots:
        return False
    registration = deps.registration_by_name.get(name)
    if registration is None:
        return False
    return not getattr(registration, "is_actor", False)


# ── Per-task trigger bookkeeping ─────────────────────────────────────────────


@dataclass
class _EdgeFlow:
    """One task's forward in-edge state for the current wave.

    Every in-edge is in exactly one of three states: *pending* (live,
    not yet fired), *fired* (delivered a trigger — optionally carrying the
    source's recorded value), or *dead* (removed from ``pending`` without
    firing). Control-readiness is ``pending`` empty + ``fired`` non-empty;
    structural death is ``pending`` and ``fired`` both empty.
    """

    pending: set[str] = field(default_factory=set)
    fired: set[str] = field(default_factory=set)
    carried: dict[str, TaskOutput] = field(default_factory=dict)


@dataclass(frozen=True)
class _Event:
    """One node completion delivered to the engine loop."""

    name: str
    raw: TaskOutput
    exc: BaseException | None
    is_parallel_fanout: bool


async def run_plan(plan: ExecutionPlan, state: WorkflowState, deps: WorkflowDeps) -> None:
    """Drive *plan* to completion, mutating *state* in place.

    Raises the first task-body exception (after cancelling in-flight
    siblings), :class:`WorkflowDeadlockError` on a structurally
    unsatisfiable graph, and the route/classification ``WorkflowError``
    subtypes exactly as the task bodies produce them.
    """
    if not plan.task_names:
        return
    engine = _PlanEngine(plan, state, deps)
    await engine.run()


class _PlanEngine:
    """Mutable per-execution scheduling state over a frozen :class:`ExecutionPlan`."""

    def __init__(self, plan: ExecutionPlan, state: WorkflowState, deps: WorkflowDeps) -> None:
        self.plan = plan
        self.state = state
        self.deps = deps
        self.flows: dict[str, _EdgeFlow] = {
            name: _EdgeFlow(pending=set(plan.in_sources.get(name, frozenset())))
            for name in plan.task_names
        }
        self.running: dict[str, asyncio.Task[None]] = {}
        self.dead: set[str] = set()
        self.events: asyncio.Queue[_Event] = asyncio.Queue()

    # ── Main loop ────────────────────────────────────────────────────────

    async def run(self) -> None:
        for entry in self.plan.entry_frontier:
            if entry in self.plan.parallel_by_body:
                continue
            self._fire(START, entry, NO_OUTPUT)
        try:
            self._launch_ready()
            while self.running:
                event = await self.events.get()
                self.running.pop(event.name, None)
                if event.exc is not None:
                    raise event.exc
                self._on_complete(event)
                self._launch_ready()
            self._check_quiescent()
        finally:
            await self._cancel_running()

    async def _cancel_running(self) -> None:
        for task in self.running.values():
            task.cancel()
        for task in self.running.values():
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.running.clear()

    # ── Edge transitions ─────────────────────────────────────────────────

    def _fire(self, src: str, tgt: str, value: TaskOutput) -> None:
        """Deliver one trigger (optionally carrying *value*) along ``src → tgt``."""
        if tgt == END_TARGET:
            return
        if (src, tgt) in self.plan.back_edges:
            # Cycle re-entry: re-launch the target directly, bypassing
            # forward coalescing. Loops are sequential through the cycle by
            # construction, so the target can never still be running here.
            extra = {src: value} if value is not NO_OUTPUT else {}
            self._launch(tgt, extra_carried=extra)
            return
        flow = self.flows[tgt]
        flow.pending.discard(src)
        flow.fired.add(src)
        if value is not NO_OUTPUT:
            flow.carried[src] = value

    def _kill(self, src: str, tgt: str) -> None:
        """Mark the edge ``src → tgt`` as never-firing; propagate task death."""
        if tgt == END_TARGET or (src, tgt) in self.plan.back_edges:
            # A dead back-edge is just a loop exit — the target already ran.
            return
        flow = self.flows[tgt]
        flow.pending.discard(src)
        if not flow.pending and not flow.fired and tgt not in self.running:
            self._mark_dead(tgt)

    def _mark_dead(self, name: str) -> None:
        """*name* can never run (all in-edges dead) — kill its out-edges too."""
        if name in self.dead:
            return
        self.dead.add(name)
        for tgt in self._targets_of(name):
            self._kill(name, tgt)
        # A dead map_over means its parallel fan-out never runs either.
        par = self.plan.parallel_by_map_over.get(name)
        if par is not None:
            self._mark_dead(par.body)

    def _targets_of(self, name: str) -> list[str]:
        edge_set = self.plan.out_edges.get(name)
        if isinstance(edge_set, BranchEdges):
            return list(edge_set.routes.values())
        if isinstance(edge_set, UnconditionalEdges):
            return list(edge_set.targets)
        return []

    # ── Readiness + launch ───────────────────────────────────────────────

    def _missing_deps(self, name: str) -> list[str]:
        registration = self.deps.registration_by_name.get(name)
        if registration is None:
            return []
        return [
            dep
            for dep in registration.depends_on
            if dep not in self.state.results and dep not in self.state.completed
        ]

    def _launch_ready(self) -> None:
        """Launch every control-ready, data-ready task; fail fast on dead deps."""
        for name in self.plan.task_names:
            if name in self.running or name in self.plan.parallel_by_body:
                continue
            flow = self.flows[name]
            if flow.pending or not flow.fired:
                continue
            missing = self._missing_deps(name)
            if missing:
                dead_missing = [dep for dep in missing if dep in self.dead]
                if dead_missing:
                    raise WorkflowDeadlockError(
                        f"task {name!r} blocked on dependencies that will "
                        f"never be satisfied: {missing} (structurally dead — "
                        f"no live path can produce {dead_missing})"
                    )
                continue  # a live upstream will record it; re-checked per event
            self._launch(name)

    def _launch(self, name: str, extra_carried: dict[str, TaskOutput] | None = None) -> None:
        flow = self.flows[name]
        carried = dict(flow.carried)
        if extra_carried:
            carried.update(extra_carried)
        # Reset the wave so a later loop iteration coalesces afresh.
        flow.pending = set(self.plan.in_sources.get(name, frozenset()))
        flow.fired = set()
        flow.carried = {}

        registration = self.deps.registration_by_name.get(name)
        delivered: TaskInput = NO_OUTPUT
        if registration is not None and not registration.depends_on and carried:
            # Values-on-edges delivery: a task with no declared data deps
            # receives the value(s) carried by its activating trigger(s) —
            # the loop-back / branch-routed input channel. The declared
            # ``depends_on`` interface always wins when present.
            if len(carried) == 1:
                delivered = next(iter(carried.values()))
            else:
                delivered = dict(carried)

        self.running[name] = asyncio.create_task(self._run_node(name, delivered))

    def _check_quiescent(self) -> None:
        """Nothing is running or launchable — any triggered-but-blocked task
        means the graph stalled; name the unsatisfiable dependencies."""
        for name in self.plan.task_names:
            if name in self.dead or name in self.plan.parallel_by_body:
                continue
            flow = self.flows[name]
            if not flow.fired:
                continue
            blockers = self._missing_deps(name) if not flow.pending else sorted(flow.pending)
            raise WorkflowDeadlockError(
                f"task {name!r} blocked on dependencies that will never be "
                f"satisfied: {blockers} (the workflow quiesced with no live "
                f"path able to produce them)"
            )

    # ── Node execution ───────────────────────────────────────────────────

    async def _run_node(self, name: str, delivered: TaskInput) -> None:
        try:
            raw = await self._invoke_body(name, delivered)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            self.events.put_nowait(_Event(name, None, exc, is_parallel_fanout=False))
            return
        self.events.put_nowait(_Event(name, raw, None, is_parallel_fanout=False))

    async def _invoke_body(self, name: str, delivered: TaskInput) -> TaskOutput:
        state, deps = self.state, self.deps
        if name in state.seeded:
            # Body already produced its value (seed_outputs); skip running it
            # but still route as if it returned that value.
            return state.results.get(name)
        mark_task_status(deps.run_dir, deps.execution_id, name, "running")
        try:
            if _cache_is_active(deps, name):
                # Opt-in content-addressed caching (batch Task only): on a hit
                # the body is skipped + cached artifacts re-registered; on a
                # miss the body runs and the result + artifact manifest are
                # stored. Routing is identical to a plain return.
                return await run_task_body_cached(name, deps, state, delivered=delivered)
            return await run_task_body(name, deps, state, delivered=delivered)
        except Exception as exc:
            mark_task_status(
                deps.run_dir,
                deps.execution_id,
                name,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    # ── Completion routing (the dispatch core) ───────────────────────────

    def _on_complete(self, event: _Event) -> None:
        name = event.name
        if event.is_parallel_fanout:
            # The fan-out already published + persisted its ordered list.
            self._fire_unconditional(
                name,
                self.plan.out_edges[name],
                self.state.results.get(name),
            )
            return

        state, deps = self.state, self.deps
        edge_set = self.plan.out_edges[name]
        try:
            recorded_value, dispatch = _classify_return(event.raw, edge_set, task_name=name)
        except Exception as exc:
            mark_task_status(
                deps.run_dir,
                deps.execution_id,
                name,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        if recorded_value is not NO_OUTPUT:
            state.record(name, recorded_value)
            mark_task_status(
                deps.run_dir,
                deps.execution_id,
                name,
                "completed",
                output=recorded_value,
                snapshot_key=_snapshot_key_of(deps, name),
            )
        else:
            state.completed.add(name)
            mark_task_status(deps.run_dir, deps.execution_id, name, "completed")

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
            # Frame-scoped End: this task's out-edges die (unless the task
            # sits on a cycle and may be re-triggered); siblings continue.
            if name not in self.plan.recurrent:
                for tgt in self._targets_of(name):
                    self._kill(name, tgt)
            return

        if isinstance(edge_set, BranchEdges):
            if not isinstance(dispatch, TakeLabel):
                # _classify_return already raises MissingRouteError for a
                # branch task returning a plain value; this arm is defensive.
                raise UnknownRouteError(
                    f"Task {name!r} has branch out-edges but produced no route."
                )
            routes = edge_set.routes
            if dispatch.label not in routes:
                raise UnknownRouteError(
                    f"Task {name!r} returned Next({dispatch.label!r}) but "
                    f"its declared routes are: {sorted(routes)}."
                )
            chosen = routes[dispatch.label]
            if name not in self.plan.recurrent:
                # A once-only branch permanently kills its non-chosen routes —
                # this is what propagates structural death to skipped legs. A
                # recurrent branch (loop until / self-loop) may fire another
                # label on a later iteration, so its edges stay live.
                for label, tgt in routes.items():
                    if label != dispatch.label:
                        self._kill(name, tgt)
            if chosen != END_TARGET:
                self._fire(name, chosen, recorded_value)
            return

        assert isinstance(edge_set, UnconditionalEdges)
        self._fire_unconditional(name, edge_set, recorded_value)

    def _fire_unconditional(self, name: str, edge_set: OutEdges, value: TaskOutput) -> None:
        targets = edge_set.targets if isinstance(edge_set, UnconditionalEdges) else ()
        for tgt in targets:
            if tgt == END_TARGET:
                continue
            par = self.plan.parallel_by_body.get(tgt)
            if par is not None and par.map_over == name:
                self._start_parallel(par)
                continue
            self._fire(name, tgt, value)

    # ── wf.parallel fan-out (engine-owned gather) ────────────────────────

    def _start_parallel(self, par: ParallelDecl) -> None:
        self.running[par.body] = asyncio.create_task(self._run_parallel(par))

    async def _run_parallel(self, par: ParallelDecl) -> None:
        try:
            await self._fan_out(par)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            self.events.put_nowait(_Event(par.body, None, exc, is_parallel_fanout=True))
            return
        self.events.put_nowait(_Event(par.body, None, None, is_parallel_fanout=True))

    async def _fan_out(self, par: ParallelDecl) -> None:
        """Run the body once per ``map_over`` element; publish the ordered list.

        Per-element failures are captured (``capture-don't-cancel``) so
        siblings finish; once every element is done, failures aggregate into
        :class:`ParallelExecutionError`. The engine owns the gather, so the
        published list is always the complete fan-out.
        """
        state, deps = self.state, self.deps
        body = par.body
        elements = list(state.results.get(par.map_over) or [])
        limiter = deps.parallel_limiters.get(body)

        async def _one(element: TaskInput) -> TaskOutput:
            mark_task_status(deps.run_dir, deps.execution_id, body, "running")
            try:
                if limiter is not None:
                    async with limiter:
                        return await run_task_body(body, deps, state, element=element)
                return await run_task_body(body, deps, state, element=element)
            except Exception as exc:  # capture per-element, aggregate below
                mark_task_status(
                    deps.run_dir,
                    deps.execution_id,
                    body,
                    "failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                return _Failure(exc)

        outcomes = await asyncio.gather(*(_one(element) for element in elements))
        failures = {
            idx: outcome.exc
            for idx, outcome in enumerate(outcomes)
            if isinstance(outcome, _Failure)
        }
        if failures:
            mark_task_status(
                deps.run_dir,
                deps.execution_id,
                body,
                "failed",
                error=f"{len(failures)} parallel element failure(s)",
            )
            raise ParallelExecutionError(body=body, failures=failures)
        ordered = list(outcomes)
        state.results[body] = ordered
        state.completed.add(body)
        state.parallel_runs[body] = len(ordered)
        mark_task_status(
            deps.run_dir,
            deps.execution_id,
            body,
            "completed",
            output=ordered,
            snapshot_key=_snapshot_key_of(deps, body),
        )


__all__ = ["run_plan"]
