"""molexp frontier scheduler — single-track edition.

``WorkflowStep`` is the **only** ``pydantic_graph.BaseNode`` molexp
exposes. pg drives ``WorkflowStep(0) → WorkflowStep(1) → … → End`` and
gets snapshot/resume bookkeeping for free; everything else (per-task
invocation, data-dep ready-set, parallel fan-out, loop counters,
``Next(label)`` routing) is molexp-owned scheduling logic that runs
inside ``WorkflowStep.run``.

``Task`` and ``Actor`` are plain abstract classes (no pg ``BaseNode``
inheritance) — the scheduler invokes ``execute(ctx)`` / ``run(ctx)``
directly via duck typing. The compiler stashes the user's registered
object straight into ``compiled.task_by_name``; there is no per-task pg
``BaseNode`` codegen.

Module surface:

* :class:`WorkflowStep` — the single pg BaseNode wrapping the scheduler.
* :func:`_classify_return` / :data:`Dispatch` — split a raw user return
  into ``(recorded_value, dispatch_verb)``.
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from mollog import get_logger
from pydantic_graph import BaseNode, End, GraphRunContext

from ..context import ActorContext
from ..protocols import (
    AssetsViewLike,
    JSONMapping,
    RunContextLike,
    Runnable,
    Streamable,
    TaskInput,
    TaskOutput,
    UserDeps,
)
from ..task import Actor, Task
from ..types import (
    BranchEdges,
    LoopMaxItersExceeded,
    MissingRouteError,
    Next,
    OutEdges,
    ParallelExecutionError,
    UnconditionalEdges,
    UnknownRouteError,
    UnknownTaskError,
    WorkflowDeadlockError,
)
from .state import WorkflowDeps, WorkflowState

if TYPE_CHECKING:
    from .._graph_decl import TaskRegistration

logger = get_logger(__name__)


# ── Dispatch sum type — runtime's view of "what to do after this task" ──────


@dataclass(frozen=True)
class TakeAll:
    """Advance to every target in this task's out-edge set."""


@dataclass(frozen=True)
class TakeLabel:
    """Advance to the target keyed by *label* (branch out-edges)."""

    label: str


@dataclass(frozen=True)
class TakeEnd:
    """Terminate the workflow at frame end (frame-scoped, §8)."""


Dispatch = TakeAll | TakeLabel | TakeEnd


class _NoOutputType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<NO_OUTPUT>"


NO_OUTPUT = _NoOutputType()


# ── End-target sentinel ─────────────────────────────────────────────────────

# Reserved string target name in ``wf.control(src, "_end")`` /
# ``wf.branch(src, routes={label: "_end"})`` declarations. The compiler
# accepts it as a target without requiring registration; the runtime
# treats reaching it as ``End()``.
END_TARGET = "_end"


# ── WorkflowStep — per-frame frontier scheduler ─────────────────────────────


@dataclass
class WorkflowStep(BaseNode[WorkflowState, WorkflowDeps, WorkflowState]):
    """One frame of the molexp frontier scheduler.

    Each ``WorkflowStep`` advances the workflow by exactly one frame:

    1. Determine the *ready set*: tasks in the current frontier (the
       initial ``entry_frontier`` for level 0, or ``state.pending_targets``
       thereafter) whose ``depends_on`` are already in
       ``state.completed``.
    2. ``asyncio.gather`` those tasks' ``run`` coroutines in parallel.
    3. Classify each task's return into ``(recorded_value, dispatch)``;
       record outputs into ``state.results``; enqueue the next-frame
       targets implied by each dispatch verb.
    4. If any task returned ``End()`` (frame-scoped) or the next frame
       is empty, return ``End(state)``; otherwise return
       ``WorkflowStep(level_index + 1)`` so pydantic-graph snapshots one
       frame per step.

    ``level_index`` is purely observability — :class:`RunStorePersistence`
    reads it to label each frame in ``workflow.json``.
    """

    level_index: int = 0

    async def run(
        self,
        ctx: GraphRunContext[WorkflowState, WorkflowDeps],
    ) -> WorkflowStep | End[WorkflowState]:
        deps = ctx.deps
        state = ctx.state

        if self.level_index == 0 and not state.pending_targets:
            frame: list[str] = list(deps.entry_frontier)
        else:
            frame = list(state.pending_targets)

        # ``Workflow.execute(seed_outputs=...)`` marks tasks as already-completed.
        # Filter them out of the frame so the body never runs; downstream
        # tasks still find their values in ``state.results``.
        if state.seeded:
            frame = [name for name in frame if name not in state.seeded]

        ready, deferred = self._partition_by_data_deps(frame, deps, state)

        if not ready:
            if (
                deferred
            ):  # pragma: no cover - defensive: compile-time UnreachableTaskError preempts this
                raise WorkflowDeadlockError(
                    f"Workflow stalled at level {self.level_index}: pending "
                    f"targets {deferred!r} have unsatisfied data dependencies "
                    "and no other task is ready to run."
                )
            return End(state)

        raw_results = await asyncio.gather(*[self._invoke_one(name, ctx) for name in ready])

        new_state, next_targets, end_signaled = self._dispatch(
            ready, raw_results, deps, new_state_seed=state, deferred=deferred
        )

        ctx.state._sync_from(new_state)

        if end_signaled or not next_targets:
            return End(ctx.state)
        return WorkflowStep(level_index=self.level_index + 1)

    @staticmethod
    async def _invoke_one(
        name: str,
        ctx: GraphRunContext[WorkflowState, WorkflowDeps],
    ) -> TaskOutput:
        """Dispatch one ready task — singleton or parallel fan-out.

        Singleton path: build a ``TaskContext`` and dispatch the user's
        registered ``Task`` / ``Actor`` / callable through
        :func:`_invoke_body_with_ctx`.

        Parallel path (``name`` is the body of a ``wf.parallel`` decl):
        read ``state.results[map_over]``, fan out one body invocation
        per element under ``Semaphore(max_concurrency)``, return the
        ordered ``list[per_element_output]``. Per-element exceptions
        are captured and raised as :class:`ParallelExecutionError`
        once every sibling has finished.
        """
        deps = ctx.deps
        registration = deps.registration_by_name.get(name)
        if registration is None:
            raise UnknownTaskError(f"WorkflowStep: unknown task {name!r} on frontier")

        decl = deps.parallel_decls.get(name) if deps.parallel_decls else None
        if decl is None:
            inputs = _collect_upstream_outputs(registration, ctx.state)
            effective_config = _resolve_dependent_params(
                registration=registration,
                state=ctx.state,
                run_context=ctx.deps.run_context,
                base_config=ctx.deps.config,
            )
            # Always construct ``ActorContext`` (a ``TaskContext`` subclass).
            # Liskov — every branch of the body dispatcher accepts
            # ``TaskContext`` ergonomically, while the Actor / Streamable
            # branches require the narrower ``ActorContext`` type for
            # ``ctx.receive()`` / ``ctx.send()`` access.
            task_ctx: ActorContext[WorkflowState, UserDeps, TaskInput] = ActorContext(
                state=ctx.state,
                deps=ctx.deps.user_deps,
                inputs=inputs,
                config=effective_config,
                run_context=ctx.deps.run_context,
            )

            # Remote-execution gate runs ahead of any local body call.
            remote = getattr(registration, "remote", None)
            if remote is not None:
                remote_executor = getattr(ctx.deps, "remote_executor", None)
                if remote_executor is not None:
                    return await remote_executor.execute_remote(
                        entry=registration,
                        inputs=task_ctx.inputs,
                        run_dir=getattr(ctx.deps, "run_dir", None),
                    )

            return await _invoke_body_with_ctx(registration, task_ctx)

        elements = ctx.state.results.get(decl.map_over)
        if elements is None:
            raise ParallelExecutionError(
                body=name,
                failures={
                    -1: RuntimeError(
                        f"wf.parallel({decl.body!r}): map_over task "
                        f"{decl.map_over!r} produced no output to fan out over."
                    )
                },
            )
        try:
            element_list = list(elements)
        except TypeError as exc:
            raise ParallelExecutionError(
                body=name,
                failures={
                    -1: TypeError(
                        f"wf.parallel({decl.body!r}): map_over task "
                        f"{decl.map_over!r} output is not iterable: {exc}"
                    )
                },
            ) from exc

        sem = asyncio.Semaphore(decl.max_concurrency)

        async def _invoke_element(elem: TaskInput) -> TaskOutput:
            async with sem:
                task_ctx: ActorContext[WorkflowState, UserDeps, TaskInput] = ActorContext(
                    state=ctx.state,
                    deps=ctx.deps.user_deps,
                    inputs=elem,
                    config=ctx.deps.config,
                    run_context=ctx.deps.run_context,
                )
                return await _invoke_body_with_ctx(registration, task_ctx)

        results_or_excs = await asyncio.gather(
            *[_invoke_element(elem) for elem in element_list],
            return_exceptions=True,
        )

        outputs: list[TaskOutput] = []
        failures: dict[int, Exception] = {}
        for idx, result in enumerate(results_or_excs):
            if isinstance(result, BaseException):
                failures[idx] = result if isinstance(result, Exception) else Exception(repr(result))
                outputs.append(None)
            else:
                outputs.append(result)

        if failures:
            raise ParallelExecutionError(body=name, failures=failures)
        return outputs

    @staticmethod
    def _partition_by_data_deps(
        frame: list[str],
        deps: WorkflowDeps,
        state: WorkflowState,
    ) -> tuple[list[str], list[str]]:
        ready: list[str] = []
        deferred: list[str] = []
        for name in frame:
            registration = deps.registration_by_name.get(name)
            if registration is None:
                raise UnknownTaskError(f"WorkflowStep: unknown task {name!r} on frontier")
            data_deps = list(getattr(registration, "depends_on", ()) or ())
            if all(d in state.completed for d in data_deps):
                ready.append(name)
            else:
                deferred.append(name)
        return ready, deferred

    @staticmethod
    def _dispatch(
        ready: list[str],
        raw_results: list[TaskOutput],
        deps: WorkflowDeps,
        new_state_seed: WorkflowState,
        deferred: list[str],
    ) -> tuple[WorkflowState, list[str], bool]:
        next_targets: list[str] = list(deferred)
        end_signaled = False
        new_state = new_state_seed

        for name, raw in zip(ready, raw_results, strict=False):
            edge_set = deps.out_edges[name]
            recorded, dispatch = _classify_return(raw, edge_set, task_name=name)
            if recorded is not NO_OUTPUT:
                new_state = new_state.record(name, recorded)
            else:
                new_state = new_state.mark_completed([name])

            # Spec 05 §4 — wf.parallel observability counter.
            # ``_invoke_one`` returns a ``list[per_element_output]`` for
            # parallel bodies; record the fan-out width once recorded.
            if name in deps.parallel_decls and isinstance(recorded, list):
                new_state = new_state.with_parallel_run(name, len(recorded))

            # Spec 04 §4 — wf.loop ``max_iters`` runtime guard. When the
            # ``until`` task dispatches Next("continue"), increment the
            # per-loop counter; once it would exceed ``max_iters``, force
            # Next("exit") and emit LoopMaxItersExceeded.
            if isinstance(dispatch, TakeLabel) and dispatch.label == "continue":
                max_iters = deps.loop_max_iters.get(name)
                if max_iters is not None:
                    new_count = new_state.loop_counters.get(name, 0) + 1
                    new_state = new_state.with_loop_counter(name, new_count)
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
                end_signaled = True
                continue
            if isinstance(dispatch, TakeLabel):
                if not isinstance(edge_set, BranchEdges):
                    raise UnknownRouteError(
                        f"Task {name!r} returned Next({dispatch.label!r}) but "
                        "has unconditional out-edges. Next() is only valid for "
                        "tasks declared with routes={...}."
                    )
                if dispatch.label not in edge_set.routes:
                    declared = sorted(edge_set.routes.keys())
                    raise UnknownRouteError(
                        f"Task {name!r} returned Next({dispatch.label!r}) but "
                        f"its declared routes are: {declared}."
                    )
                target = edge_set.routes[dispatch.label]
                if target == END_TARGET:
                    end_signaled = True
                else:
                    next_targets.append(target)
            elif isinstance(dispatch, TakeAll):
                if isinstance(edge_set, UnconditionalEdges):
                    for tgt in edge_set.targets:
                        if tgt == END_TARGET:
                            end_signaled = True
                        else:
                            next_targets.append(tgt)

        seen: set[str] = set()
        deduped: list[str] = []
        for t in next_targets:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        new_state = new_state.set_pending(deduped)
        return new_state, deduped, end_signaled


# ── Return-value classifier (spec 03 §5 / §9 step 1) ────────────────────────


def _classify_return(
    value: TaskOutput,
    edge_set: OutEdges,
    *,
    task_name: str,
) -> tuple[TaskOutput, Dispatch]:
    """Split a raw task return value into ``(recorded_value, dispatch_verb)``.

    Spec 03 §5 return shapes:

    * ``Output`` / ``None`` → ``(value, TakeAll)``
    * ``Next(label)`` → ``(NO_OUTPUT, TakeLabel(label))``
    * ``End()`` → ``(NO_OUTPUT, TakeEnd)``
    * ``(val, Next(label))`` → ``(val, TakeLabel(label))``
    * ``(val, End())`` → ``(val, TakeEnd)``

    A :class:`BranchEdges` task returning a plain value (no ``Next`` / ``End``)
    raises :class:`MissingRouteError` listing the declared labels.
    """
    if isinstance(value, tuple) and len(value) == 2:
        v, sentinel = value
        if isinstance(sentinel, Next):
            return v, TakeLabel(sentinel.label)
        if isinstance(sentinel, End):
            return v, TakeEnd()
        # Otherwise fall through — a 2-tuple is just a value.

    if isinstance(value, Next):
        return NO_OUTPUT, TakeLabel(value.label)
    if isinstance(value, End):
        return NO_OUTPUT, TakeEnd()

    if isinstance(edge_set, BranchEdges):
        labels = sorted(edge_set.routes.keys())
        raise MissingRouteError(
            f"Task '{task_name}' has branch out-edges (declared labels: {labels}) "
            f"but its body returned a plain value without `Next(label)` or `End()`. "
            f"Either return Next(label) selecting one of {labels}, return End() to "
            f"terminate, or convert the task to unconditional out-edges."
        )

    return value, TakeAll()


# ── Body dispatcher (called by WorkflowStep._invoke_one) ────────────────────


async def run_task_body(
    name: str,
    deps: WorkflowDeps,
    state: WorkflowState,
    *,
    element: TaskInput = NO_OUTPUT,
) -> TaskOutput:
    """Invoke one task's body against freshly-built context (pg-node edition).

    Extracted from the old ``WorkflowStep._invoke_one`` singleton path so the
    per-task pydantic-graph ``Step`` nodes (see :mod:`.lowering`) reuse the
    exact context-construction + remote-gate + dependent-params logic. When
    *element* is provided (``wf.parallel`` fan-out), it is used as ``ctx.inputs``
    directly instead of collecting upstream outputs.
    """
    registration = deps.registration_by_name.get(name)
    if registration is None:
        raise UnknownTaskError(f"run_task_body: unknown task {name!r}")

    if element is not NO_OUTPUT:
        inputs: TaskInput = element
        effective_config = deps.config
    else:
        inputs = _collect_upstream_outputs(registration, state)
        effective_config = _resolve_dependent_params(
            registration=registration,
            state=state,
            run_context=deps.run_context,
            base_config=deps.config,
        )

    task_ctx: ActorContext[WorkflowState, UserDeps, TaskInput] = ActorContext(
        state=state,
        deps=deps.user_deps,
        inputs=inputs,
        config=effective_config,
        run_context=deps.run_context,
    )

    remote = getattr(registration, "remote", None)
    if remote is not None and element is NO_OUTPUT:
        remote_executor = getattr(deps, "remote_executor", None)
        if remote_executor is not None:
            return await remote_executor.execute_remote(
                entry=registration,
                inputs=task_ctx.inputs,
                run_dir=getattr(deps, "run_dir", None),
            )

    return await _invoke_body_with_ctx(registration, task_ctx)


async def run_parallel_body(
    name: str, deps: WorkflowDeps, state: WorkflowState
) -> list[TaskOutput]:
    """Fan out a ``wf.parallel`` body over ``map_over``'s output (ordered, bounded).

    Mirrors the old ``WorkflowStep._invoke_one`` parallel path: one body
    invocation per element under ``Semaphore(max_concurrency)``, ordered
    results, per-element exceptions aggregated into
    :class:`ParallelExecutionError`.
    """
    decl = deps.parallel_decls[name]
    elements = state.results.get(decl.map_over)
    if elements is None:
        raise ParallelExecutionError(
            body=name,
            failures={
                -1: RuntimeError(
                    f"wf.parallel({decl.body!r}): map_over task "
                    f"{decl.map_over!r} produced no output to fan out over."
                )
            },
        )
    try:
        element_list = list(elements)
    except TypeError as exc:
        raise ParallelExecutionError(
            body=name,
            failures={
                -1: TypeError(
                    f"wf.parallel({decl.body!r}): map_over task "
                    f"{decl.map_over!r} output is not iterable: {exc}"
                )
            },
        ) from exc

    sem = asyncio.Semaphore(decl.max_concurrency)

    async def _invoke_element(elem: TaskInput) -> TaskOutput:
        async with sem:
            return await run_task_body(name, deps, state, element=elem)

    results_or_excs = await asyncio.gather(
        *[_invoke_element(elem) for elem in element_list],
        return_exceptions=True,
    )
    outputs: list[TaskOutput] = []
    failures: dict[int, Exception] = {}
    for idx, result in enumerate(results_or_excs):
        if isinstance(result, BaseException):
            failures[idx] = result if isinstance(result, Exception) else Exception(repr(result))
            outputs.append(None)
        else:
            outputs.append(result)
    if failures:
        raise ParallelExecutionError(body=name, failures=failures)
    return outputs


async def _invoke_body_with_ctx(
    registration: TaskRegistration,
    task_ctx: ActorContext[TaskOutput, UserDeps, TaskInput],
) -> TaskOutput:
    """Dispatch a registered task's body against a *pre-built* TaskContext.

    Single-track edition: ``registration.fn_or_class`` is the user-supplied
    object (Task / Actor instance, third-party Runnable / Streamable, or
    plain callable). No per-task pg ``BaseNode`` codegen, no patched
    ``Task.run`` — this function IS the body dispatcher.
    """
    body = registration.fn_or_class

    # OOP Task subclass — invoke .execute(ctx).
    if isinstance(body, Task):
        return await body.execute(task_ctx)

    # OOP Actor subclass — drain the async generator.
    if isinstance(body, Actor):
        last: TaskOutput = None
        async for chunk in body.run(task_ctx):
            last = chunk
        return last

    # Third-party Runnable (anything with .execute) — protocol path.
    if isinstance(body, Runnable):
        return await body.execute(task_ctx)

    # Third-party Streamable (anything with .run async generator).
    if isinstance(body, Streamable):
        last2: TaskOutput = None
        async for chunk in body.run(task_ctx):
            last2 = chunk
        return last2

    # Decorator-actor: async-generator function. ``is_actor=True`` is the
    # registry's contract that ``body(ctx)`` returns an ``AsyncIterator``;
    # the cast tells the static type-checker which arm of ``TaskBody`` we
    # are in (the runtime check above is the actual guard).
    if getattr(registration, "is_actor", False) and callable(body):
        actor_fn = cast("Callable[..., AsyncIterator[TaskOutput]]", body)
        last3: TaskOutput = None
        async for chunk in actor_fn(task_ctx):
            last3 = chunk
        return last3

    # Decorator-task: plain async function.
    if callable(body):
        return await cast("Callable[..., Awaitable[TaskOutput]]", body)(task_ctx)

    raise TypeError(
        f"Task '{registration.name}' is neither Task / Actor / Runnable / "
        f"Streamable / callable: {type(body)}"
    )


# ── Helpers ─────────────────────────────────────────────────────────────────


def _collect_upstream_outputs(registration: TaskRegistration, state: WorkflowState) -> TaskInput:
    """Collect upstream outputs into the shape ``TaskContext.inputs`` expects.

    Returns ``None`` for no deps; the single value for one dep; a
    ``dict[name → value]`` for multiple deps.
    """
    deps = list(registration.depends_on)
    if not deps:
        return None
    if len(deps) == 1:
        return state.results.get(deps[0])
    return {dep: state.results.get(dep) for dep in deps}


class _UpstreamView:
    """Per-upstream view passed to ``dependent_params(prev)``.

    Exposes ``.output`` (the upstream task's return value, as recorded in
    :attr:`WorkflowState.results`) and ``.assets`` (an
    :class:`~molexp.workspace.assets.AssetsView` filtered to the upstream
    task's producer entries when a workspace ``RunContext`` is attached;
    ``None`` otherwise).
    """

    __slots__ = ("assets", "output")

    def __init__(self, output: TaskOutput, assets: _UpstreamAssetsView | None) -> None:
        self.output = output
        self.assets = assets


def _resolve_dependent_params(
    *,
    registration: TaskRegistration,
    state: WorkflowState,
    run_context: RunContextLike | None,
    base_config: JSONMapping | None,
) -> JSONMapping | None:
    """If the task declares ``dependent_params=fn``, resolve and overlay onto config.

    ``fn`` receives ``dict[str, _UpstreamView]`` keyed by upstream task name.
    Its return mapping is overlayed onto a fresh
    :class:`~molexp.profile.ProfileConfig` and the result replaces the task's
    base config. The base config is returned unchanged when no
    ``dependent_params`` is declared.
    """
    fn = getattr(registration, "dependent_params", None)
    if fn is None:
        return base_config

    from molexp.profile import ProfileConfig

    prev: dict[str, _UpstreamView] = {}
    for dep in registration.depends_on:
        upstream_assets = None
        if run_context is not None:
            assets_view = getattr(run_context, "assets", None)
            if assets_view is not None and hasattr(assets_view, "query"):
                upstream_assets = _UpstreamAssetsView(assets_view, producer_task=dep)
        prev[dep] = _UpstreamView(
            output=state.results.get(dep),
            assets=upstream_assets,
        )

    overlay = fn(prev)
    if overlay is None:
        return base_config
    if not isinstance(overlay, Mapping):
        raise TypeError(
            f"dependent_params for task {registration.name!r} must return a Mapping; "
            f"got {type(overlay).__name__}"
        )
    merged: dict[str, TaskInput] = dict(base_config) if base_config is not None else {}
    merged.update(overlay)
    return ProfileConfig(merged, name=getattr(base_config, "name", None))


class _UpstreamAssetsView:
    """Lazy ``query()`` proxy that pre-binds ``producer_task=<dep>``.

    Avoids importing :class:`AssetsView` at module top to keep the
    workspace dependency optional for non-workspace runs.
    """

    __slots__ = ("_inner", "_producer_task")

    def __init__(self, assets_view: AssetsViewLike, producer_task: str) -> None:
        self._inner = assets_view
        self._producer_task = producer_task

    def query(
        self,
        *,
        kind: str | type | None = None,
        producer_run: str | None = None,
        producer_task: str | None = None,
        tag: tuple[str, str] | None = None,
        limit: int | None = None,
        recursive: bool = False,
    ) -> TaskOutput:
        return self._inner.query(
            kind=kind,
            producer_run=producer_run,
            producer_task=producer_task or self._producer_task,
            tag=tag,
            limit=limit,
            recursive=recursive,
        )

    def list(self) -> TaskOutput:
        return self.query()


__all__ = [
    "END_TARGET",
    "NO_OUTPUT",
    "Dispatch",
    "End",
    "TakeAll",
    "TakeEnd",
    "TakeLabel",
    "WorkflowStep",
    "_classify_return",
]
