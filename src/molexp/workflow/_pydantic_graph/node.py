"""molexp per-task pydantic-graph lowering — node-body helpers.

The workflow DAG is lowered to a genuine ``pydantic_graph`` graph with
**one Step per task** (see :mod:`.compiler`). pydantic-graph primitives
carry control flow — edges for data/control deps, ``Join`` for
multi-dependency fan-in, map-Fork + ``Join`` for ``wf.parallel``,
``Decision`` for ``wf.branch`` / ``wf.loop`` routing.

molexp tasks do **not** read inputs from edge tokens — each task reads
its upstream outputs from the shared, mutated :class:`WorkflowState`
``results`` dict. Edges express TRIGGER / ORDERING only. The token value
matters only for (a) ``wf.parallel`` map fan-out (the list to spread) and
(b) branch routing (the ``Next`` token fed to a ``Decision``).

``Task`` and ``Actor`` are plain abstract classes (no pg ``BaseNode``
inheritance) — the Step body invokes ``execute(ctx)`` / ``run(ctx)``
directly via duck typing.

Module surface:

* :func:`run_task_body` — invoke one task's body against a fresh
  ``TaskContext`` (reused by every Step factory).
* :func:`_classify_return` / :data:`Dispatch` — split a raw user return
  into ``(recorded_value, dispatch_verb)``.
* :class:`_Failure` — wraps a captured per-element parallel exception.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic_graph import End

from ..context import TaskContext
from ..protocols import (
    Runnable,
    Streamable,
    TaskInput,
    TaskOutput,
    UserDeps,
)
from ..task import Actor, Task
from ..types import (
    BranchEdges,
    MissingRouteError,
    MissingUpstreamResultError,
    Next,
    OutEdges,
    UnknownTaskError,
)
from .node_params import _resolve_dependent_params
from .state import WorkflowDeps, WorkflowState

if TYPE_CHECKING:
    from .._graph_decl import TaskRegistration


# ── Dispatch sum type — "what to do after this task" ────────────────────────


@dataclass(frozen=True)
class TakeAll:
    """Advance to every target in this task's out-edge set."""


@dataclass(frozen=True)
class TakeLabel:
    """Advance to the target keyed by *label* (branch out-edges)."""

    label: str


@dataclass(frozen=True)
class TakeEnd:
    """Terminate the workflow."""


Dispatch = TakeAll | TakeLabel | TakeEnd


@dataclass
class _Failure:
    """Wraps a captured exception from one ``wf.parallel`` element.

    The map-Fork body wrapper returns ``{idx: _Failure(exc)}`` instead of
    raising, so a single failing element never cancels its siblings; the
    collector step aggregates failures into :class:`ParallelExecutionError`
    once every element has finished (capture-don't-cancel, ac-004).
    """

    exc: Exception


@dataclass(frozen=True)
class _Trigger:
    """Routing token: advance to this Step's unconditional out-edge targets.

    Returned by an unconditional Step and matched by the per-Step
    ``Decision`` to fan out to every target (or to ``end_node`` for a
    terminal Step). The token carries no payload — molexp tasks read their
    inputs from the shared ``state.results``, never from edge tokens.
    """


@dataclass(frozen=True)
class _EndTok:
    """Routing token: terminate the workflow at this Step.

    Returned when a Step's body yields ``End(...)`` (bare or as the second
    element of a ``(value, End())`` tuple) or routes to the ``_end`` target.
    The per-Step ``Decision`` matches it and routes to ``end_node``.
    """


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


# ── Body dispatcher (called by every per-task Step) ─────────────────────────


async def run_task_body(
    name: str,
    deps: WorkflowDeps,
    state: WorkflowState,
    *,
    element: TaskInput = NO_OUTPUT,
) -> TaskOutput:
    """Invoke one task's body against a freshly-built context.

    The per-task pydantic-graph ``Step`` nodes (built in :mod:`.compiler`)
    all route through here so context-construction + remote-gate +
    dependent-params logic lives in one place. When *element* is provided
    (``wf.parallel`` fan-out), it is used as ``ctx.inputs`` directly instead
    of collecting upstream outputs.
    """
    registration = deps.registration_by_name.get(name)
    if registration is None:
        raise UnknownTaskError(f"run_task_body: unknown task {name!r}")

    # Mark this body in flight for the whole duration it executes. The
    # dependency barrier in :mod:`.compiler` reads ``state.running`` as a
    # frontier-liveness signal: while any body is running it cannot declare
    # a deadlock, so a slow upstream is never mistaken for an absent one.
    state.running += 1
    try:
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

        task_ctx: TaskContext[WorkflowState, UserDeps, TaskInput] = TaskContext(
            state=state,
            deps=deps.user_deps,
            inputs=inputs,
            config=effective_config,
            run_context=deps.run_context,
        )

        # Tag artifacts produced by this body with ``producer.task_id == name``
        # so the cache hook (and asset queries) can find them by producer task.
        set_active_task = getattr(deps.run_context, "set_active_task", None)
        if callable(set_active_task):
            set_active_task(name)

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
    finally:
        state.running -= 1
        # Wake any barrier waiter so it re-evaluates the frontier the instant a
        # body finishes (the last one finishing is how a deadlock is detected).
        state.signal_progress()


async def _invoke_body_with_ctx(
    registration: TaskRegistration,
    task_ctx: TaskContext[TaskOutput, UserDeps, TaskInput],
) -> TaskOutput:
    """Dispatch a registered task's body against a *pre-built* TaskContext.

    ``registration.fn_or_class`` is the user-supplied object (Task / Actor
    instance, third-party Runnable / Streamable, or plain callable). No
    per-task pg ``BaseNode`` codegen, no patched ``Task.run`` — this
    function IS the body dispatcher.
    """
    body = registration.fn_or_class

    # Tag any asset saved during this task with its name as ``Producer.task_id``
    # via the run_context's active-task slot (the slot is read by ArtifactAccessor
    # on write). molexp task bodies run inline/blocking on the event-loop thread,
    # so a single slot is effectively per-task here; the next task overwrites it
    # before saving its own assets.
    _rc = getattr(task_ctx, "run_context", None)
    if _rc is not None and hasattr(_rc, "set_active_task"):
        _rc.set_active_task(registration.name)

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
    # Fail fast on a declared dependency that never ran instead of silently
    # coalescing to ``None`` (the old ``dict.get`` behavior, which delivered
    # ``None`` to a parallel-join consumer). A dep is satisfied if it recorded a
    # result OR completed without one (a branch/routing task returns ``Next``
    # and lands in ``completed`` but not ``results`` — its value is legitimately
    # ``None``). The barrier guarantees presence on the happy path, so a dep in
    # neither set is a genuine never-ran error, not a silent ``None``.
    missing = [dep for dep in deps if dep not in state.results and dep not in state.completed]
    if missing:
        raise MissingUpstreamResultError(registration.name, missing, sorted(state.results))
    if len(deps) == 1:
        return state.results.get(deps[0])
    return {dep: state.results.get(dep) for dep in deps}


__all__ = [
    "END_TARGET",
    "NO_OUTPUT",
    "Dispatch",
    "End",
    "TakeAll",
    "TakeEnd",
    "TakeLabel",
    "_EndTok",
    "_Failure",
    "_Trigger",
    "_classify_return",
    "run_task_body",
]
