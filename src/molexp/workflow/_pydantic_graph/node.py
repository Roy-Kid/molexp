"""Per-task BaseNode wrappers + return-value classifier (spec 03 §9, §10).

``Task`` already inherits :class:`pydantic_graph.BaseNode` (see
``workflow/task.py``), so each registered task is a BaseNode subclass
straight away. The compiler creates a thin per-registration subclass via
``type(...)`` to attach the task name + registration as class attributes;
no method codegen, no closures.

The molexp frontier runtime drives ``await node.run(ctx)`` directly and
classifies the raw return value via :func:`_classify_return`. ``Graph.run``
is never invoked (spec 03 §9), so the BaseNode contract on ``run``'s return
type is purely satisfied for ``Graph(nodes=[...])`` construction; runtime
correctness lives in the scheduler.

Module surface:

* :class:`_CallableTask` — fixed Task base for decorator-style
  ``@wf.task async def`` functions.
* :class:`_StreamableTask` — fixed Actor base for ``@wf.actor`` async
  generators (drains the generator; terminal yield is the dispatch value).
* :func:`make_task_node_class` — return a per-registration subclass with
  ``_molexp_task_name`` / ``_molexp_registration`` set.
* :func:`_classify_return` / :data:`Dispatch` — split a raw user return
  into ``(recorded_value, dispatch_verb)``.
"""

from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass
from typing import Any

from mollog import get_logger
from pydantic_graph import BaseNode, End, GraphRunContext

from ..context import TaskContext
from ..protocols import Runnable, Streamable
from ..task import Task
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
from ..types import (
    End as MolExpEnd,
)
from .state import WorkflowDeps, WorkflowState

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
    ) -> "WorkflowStep | End[WorkflowState]":
        deps = ctx.deps
        state = ctx.state

        if self.level_index == 0 and not state.pending_targets:
            frame: list[str] = list(deps.entry_frontier)
        else:
            frame = list(state.pending_targets)

        ready, deferred = self._partition_by_data_deps(frame, deps, state)

        if not ready:
            if deferred:
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
    ) -> Any:
        """Dispatch one ready task — singleton or parallel fan-out.

        Singleton path: ``node.run(ctx)`` directly, which routes through
        the patched :func:`_task_run` and returns the user's raw value.

        Parallel path (``name`` is the body of a ``wf.parallel`` decl):
        read ``state.results[map_over]``, fan out one body invocation
        per element under ``Semaphore(max_concurrency)``, return the
        ordered ``list[per_element_output]``. Per-element exceptions
        are captured and raised as :class:`ParallelExecutionError`
        once every sibling has finished (spec 05 §4 D3).
        """
        deps = ctx.deps
        node = deps.task_by_name[name]
        decl = deps.parallel_decls.get(name) if deps.parallel_decls else None
        if decl is None:
            return await node.run(ctx)  # type: ignore[arg-type]

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
        registration = getattr(type(node), "_molexp_registration", None) or getattr(
            node, "_molexp_registration", None
        )

        async def _invoke_element(elem: Any) -> Any:
            async with sem:
                task_ctx = TaskContext(
                    state=ctx.state,
                    deps=ctx.deps.user_deps,
                    inputs=elem,
                    config=ctx.deps.config,
                    run_context=ctx.deps.run_context,
                )
                return await _invoke_body_with_ctx(node, registration, task_ctx)

        results_or_excs = await asyncio.gather(
            *[_invoke_element(elem) for elem in element_list],
            return_exceptions=True,
        )

        outputs: list[Any] = []
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
            node = deps.task_by_name.get(name)
            if node is None:
                raise UnknownTaskError(f"WorkflowStep: unknown task {name!r} on frontier")
            data_deps = _node_depends_on(node)
            if all(d in state.completed for d in data_deps):
                ready.append(name)
            else:
                deferred.append(name)
        return ready, deferred

    @staticmethod
    def _dispatch(
        ready: list[str],
        raw_results: list[Any],
        deps: WorkflowDeps,
        new_state_seed: WorkflowState,
        deferred: list[str],
    ) -> tuple[WorkflowState, list[str], bool]:
        next_targets: list[str] = list(deferred)
        end_signaled = False
        new_state = new_state_seed

        for name, raw in zip(ready, raw_results):
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


def _node_depends_on(node: Any) -> list[str]:
    """Pull the data ``depends_on`` list off a per-task BaseNode subclass.

    The compiler stashes the originating :class:`TaskRegistration` on
    each generated subclass via ``_molexp_registration``; absent a
    registration (placeholder / sentinel nodes), the node is treated
    as having no data deps.
    """
    registration = getattr(type(node), "_molexp_registration", None)
    if registration is None:
        return []
    return list(registration.depends_on)


# ── Return-value classifier (spec 03 §5 / §9 step 1) ────────────────────────


def _classify_return(
    value: Any,
    edge_set: OutEdges,
    *,
    task_name: str,
) -> tuple[Any, Dispatch]:
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
        if isinstance(sentinel, MolExpEnd):
            return v, TakeEnd()
        # Otherwise fall through — a 2-tuple is just a value.

    if isinstance(value, Next):
        return NO_OUTPUT, TakeLabel(value.label)
    if isinstance(value, MolExpEnd):
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


# ── Fixed Task subclasses for non-OOP registrations ─────────────────────────


class _CallableTask(Task[Any, Any, Any, Any]):
    """Wraps a decorator-style ``@wf.task async def fetch(ctx)``.

    Per-registration subclass attaches ``_molexp_fn`` (the user's function)
    and ``_molexp_task_name`` (the registered name).
    """

    _molexp_fn: Any = None  # set on per-registration subclass

    async def execute(self, ctx: TaskContext[Any, Any, Any]) -> Any:
        fn = type(self)._molexp_fn
        if fn is None:
            raise RuntimeError(f"{type(self).__name__}: _molexp_fn was not bound at compile time")
        return await fn(ctx)


class _StreamableTask(Task[Any, Any, Any, Any]):
    """Wraps a decorator-style ``@wf.actor async def streamer(ctx)`` async generator.

    Drains the generator; the terminal yield is returned as the dispatch value.
    """

    _molexp_fn: Any = None

    async def execute(self, ctx: TaskContext[Any, Any, Any]) -> Any:
        fn = type(self)._molexp_fn
        if fn is None:
            raise RuntimeError(f"{type(self).__name__}: _molexp_fn was not bound at compile time")
        last: Any = None
        async for chunk in fn(ctx):
            last = chunk
        return last


# ── Concrete Task.run via mixin (overrides the placeholder in task.py) ──────
# Done as a method patch so user-facing Task.py stays small / dependency-free.


async def _task_run(
    self: Task[Any, Any, Any, Any],
    ctx: GraphRunContext[WorkflowState, WorkflowDeps],
) -> Any:
    """Concrete BaseNode.run for Task: build TaskContext, dispatch, return raw value.

    The molexp frontier runtime classifies the raw return via
    :func:`_classify_return`. The annotation on the BaseNode contract
    (``BaseNode | End``) is only relevant for ``Graph.run``, which we don't
    invoke. Type-wise this is intentionally loose.
    """
    # The compiler may stash ``_molexp_registration`` either on the
    # per-registration subclass (decorator-style — fresh ``cls()``
    # instances) or directly on the user's instance (OOP-style — we
    # reuse their ``Task`` instance because its ``__init__`` may
    # require args we don't have). ``getattr(self, ...)`` finds both.
    registration = getattr(self, "_molexp_registration", None)
    if registration is None:
        raise RuntimeError(
            f"{type(self).__name__}.run() called without registration metadata. "
            "Tasks must go through Workflow.add(...) / @wf.task before execution."
        )

    inputs = _collect_upstream_outputs(registration, ctx.state)
    task_ctx = TaskContext(
        state=ctx.state,
        deps=ctx.deps.user_deps,
        inputs=inputs,
        config=ctx.deps.config,
        run_context=ctx.deps.run_context,
    )

    # Remote-execution gate runs ahead of any local body call.
    remote = getattr(registration, "remote", None)
    if remote is not None:
        remote_executor = getattr(ctx.deps, "remote_executor", None)
        if remote_executor is not None:
            run_dir = getattr(ctx.deps, "run_dir", None)
            return await remote_executor.execute_remote(
                entry=registration,
                inputs=task_ctx.inputs,
                run_dir=run_dir,
            )

    return await _invoke_body_with_ctx(self, registration, task_ctx)


async def _invoke_body_with_ctx(
    node: Any,
    registration: Any,
    task_ctx: TaskContext[Any, Any, Any],
) -> Any:
    """Dispatch a registered task's body against a *pre-built* TaskContext.

    Mirrors the body-dispatch logic from :func:`_task_run` but takes a
    caller-built ``TaskContext`` so the parallel scheduler can supply
    per-element ``inputs`` instead of going through
    :func:`_collect_upstream_outputs`.
    """
    body = registration.fn_or_class

    # OOP-style: the registered object IS this Task instance subclass; call its execute.
    if isinstance(body, Task):
        return await body.execute(task_ctx)

    # Third-party Runnable (anything with .execute) — protocol path.
    if isinstance(body, Runnable):
        return await body.execute(task_ctx)

    # Streamable async-generator (third-party actor) — drain.
    if isinstance(body, Streamable):
        last: Any = None
        async for chunk in body.run(task_ctx):
            last = chunk
        return last

    # Decorator-style: the per-registration subclass binds ``_molexp_fn``.
    bound_fn = getattr(type(node), "_molexp_fn", None)
    if bound_fn is not None:
        return await node.execute(task_ctx)

    # Bare callable (function) attached as fn_or_class but no _molexp_fn.
    if callable(body):
        return await body(task_ctx)

    raise TypeError(
        f"Task '{registration.name}' is neither Task / Runnable / Streamable / callable: "
        f"{type(body)}"
    )


# Patch Task with the concrete run; BaseNode.run is abstract there until now.
Task.run = _task_run  # type: ignore[assignment, method-assign]


# ── make_task_node_class — thin per-registration subclass ───────────────────


def make_task_node_class(
    *,
    name: str,
    registration: Any,
    edge_set: OutEdges,
) -> type[Task[Any, Any, Any, Any]]:
    """Return a per-registration Task subclass with ``_molexp_task_name`` set.

    OOP-style (``registration.fn_or_class`` is a :class:`Task` instance) →
    subclass *its concrete class* so the user's ``execute`` is reused.

    Decorator-style (callable) → subclass :class:`_CallableTask`
    (or :class:`_StreamableTask` for actors) and bind ``_molexp_fn`` to the
    user's function on the resulting class.
    """
    safe = _python_safe_ident(name)
    cls_name = f"_GraphNode_{safe}"

    body = registration.fn_or_class
    namespace: dict[str, Any] = {
        "_molexp_task_name": name,
        "_molexp_registration": registration,
        "_molexp_edge_set": edge_set,
        "__module__": __name__,
    }

    if isinstance(body, Task):
        # OOP-style: subclass the user's concrete Task subclass.
        base: type[Task[Any, Any, Any, Any]] = type(body)
    elif registration.is_actor:
        # Decorator actor — bind the async-generator function.
        base = _StreamableTask
        namespace["_molexp_fn"] = body
    else:
        # Decorator function or any callable — bind it.
        base = _CallableTask
        namespace["_molexp_fn"] = body

    cls = type(cls_name, (base,), namespace)
    # Sanity: subclass must be a BaseNode for Graph(nodes=[...]) to accept it.
    assert issubclass(cls, BaseNode), f"{cls_name} is not a BaseNode subclass"
    return cls  # type: ignore[return-value]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _collect_upstream_outputs(registration: Any, state: WorkflowState) -> Any:
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


def _python_safe_ident(name: str) -> str:
    """Sanitise a task name so it can serve as part of a Python class name."""
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe or "_anonymous"


__all__ = [
    "END_TARGET",
    "End",
    "Dispatch",
    "TakeAll",
    "TakeLabel",
    "TakeEnd",
    "NO_OUTPUT",
    "WorkflowStep",
    "_classify_return",
    "_CallableTask",
    "_StreamableTask",
    "make_task_node_class",
]
