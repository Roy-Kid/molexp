"""molexp per-task node-body helpers for the structural engine.

The workflow DAG is lowered to an :class:`~.plan.ExecutionPlan` (see
:mod:`.compiler`) and driven by the values-on-edges engine in
:mod:`.engine`: each task's inputs are delivered from its upstreams'
outputs — via the declared ``depends_on`` interface, the engine-injected
``root_inputs``, or the value carried on the activating trigger edge
(branch-routed / loop-back delivery).

``Task`` and ``Actor`` are plain abstract classes (no pg ``BaseNode``
inheritance) — the engine invokes ``execute(ctx)`` / ``run(ctx)``
directly via duck typing. ``End`` is the re-exported
``pydantic_graph.End`` sentinel (the layer's remaining pg surface).

Module surface:

* :func:`run_task_body` — invoke one task's body against a fresh
  ``TaskContext`` (reused by the engine for every node).
* :func:`_classify_return` / :data:`Dispatch` — split a raw user return
  into ``(recorded_value, dispatch_verb)``.
* :class:`_Failure` — wraps a captured per-element parallel exception.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic_graph import End

from ..context import TaskContext
from ..protocols import (
    Runnable,
    Streamable,
    TaskInput,
    TaskOutput,
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
    from ..compiled import CompiledWorkflow
    from ..protocols import JSONMapping


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


def _merge_delivered(base: TaskInput, delivered: TaskInput) -> TaskInput:
    """Fold a trigger-delivered value into engine-injected root inputs.

    Mirrors the SubWorkflow ``root_input`` forwarding merge: when both are
    dicts they MERGE (delivered keys win) so a loop-back / branch-routed value
    reaches a workspace root task without losing ``params`` / ``workdir``;
    otherwise the delivered value replaces the base.
    """
    if isinstance(base, dict) and isinstance(delivered, dict):
        return {**base, **delivered}
    return delivered


async def run_task_body(
    name: str,
    deps: WorkflowDeps,
    state: WorkflowState,
    *,
    element: TaskInput = NO_OUTPUT,
    delivered: TaskInput = NO_OUTPUT,
) -> TaskOutput:
    """Invoke one task's body against a freshly-built context.

    The engine (:mod:`.engine`) routes every node through here so
    context-construction + remote-gate + dependent-params logic lives in one
    place. Input resolution (values-on-edges):

    * *element* (``wf.parallel`` fan-out) is used as ``ctx.inputs`` directly;
    * engine-injected ``root_inputs`` (run params + content-addressed
      workdir) come next, merged with *delivered* when one was carried;
    * a non-empty ``depends_on`` collects the declared upstream outputs;
    * otherwise *delivered* — the value carried on the activating trigger
      edge (a branch-routed value or a loop-back from the previous
      iteration) — becomes ``ctx.inputs``.
    """
    registration = deps.registration_by_name.get(name)
    if registration is None:
        raise UnknownTaskError(f"run_task_body: unknown task {name!r}")

    body = registration.fn_or_class
    if element is not NO_OUTPUT:
        inputs: TaskInput = element
        effective_config = deps.config
    else:
        if name in state.root_inputs:
            # Engine-injected root inputs (sweep params + content-addressed
            # workdir Path). The body still runs; only its inputs are pre-set.
            # A trigger-delivered value (loop-back into a root task) merges in.
            inputs = state.root_inputs[name]
            if delivered is not NO_OUTPUT:
                inputs = _merge_delivered(inputs, delivered)
        elif registration.depends_on:
            inputs = _collect_upstream_outputs(registration, state)
        elif delivered is not NO_OUTPUT:
            # Values-on-edges: the activating edge carried the upstream's
            # recorded output (branch-routed value / loop-back iteration).
            inputs = delivered
        else:
            inputs = _collect_upstream_outputs(registration, state)
        effective_config = _resolve_dependent_params(
            registration=registration,
            state=state,
            run_context=deps.run_context,
            base_config=deps.config,
        )

    # Capabilities-as-inputs: an engine-internal task may declare
    # ``__wf_capability__``; the engine injects the named capability as
    # ``ctx.inputs`` so the task stays on the pure {inputs, config} contract
    # (no ``run_context``). ``sub_runner`` runs an inner workflow bound to
    # this run via the PRIVATE ``deps.run_context`` channel.
    if getattr(body, "__wf_capability__", None) == "sub_runner":
        # The node's resolved input (fan-out element / upstream output / routed
        # value / root params) is forwarded into the inner workflow's entry task;
        # the closure replaces ``ctx.inputs`` so SubWorkflow.execute stays on the
        # pure {inputs=sub_runner, config} contract. Only forward when the node
        # actually has an input — a bare root SubWorkflow (no element, no deps,
        # no workspace root params) runs the inner unchanged, so an inner spec
        # with several roots is not forced to declare a single entry.
        has_input = (
            element is not NO_OUTPUT
            or delivered is not NO_OUTPUT
            or name in state.root_inputs
            or bool(registration.depends_on)
        )
        inputs = _make_sub_runner(deps, root_input=inputs, forward=has_input)

    task_ctx: TaskContext[Any, TaskInput] = TaskContext(
        inputs=inputs,
        config=effective_config,
        state=state,
        workdir=_workdir_for(deps, name),
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


async def _invoke_body_with_ctx(
    registration: TaskRegistration,
    task_ctx: TaskContext[Any, TaskInput],
) -> TaskOutput:
    """Dispatch a registered task's body against a *pre-built* TaskContext.

    ``registration.fn_or_class`` is the user-supplied object (Task / Actor
    instance, third-party Runnable / Streamable, or plain callable). No
    per-task pg ``BaseNode`` codegen, no patched ``Task.run`` — this
    function IS the body dispatcher. The producer-task tag is set once by the
    caller (:func:`run_task_body`) via the private ``deps.run_context``.
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


def _workdir_for(deps: WorkflowDeps, name: str):  # noqa: ANN202
    """Content-addressed scratch ``Path`` for task *name* (``None`` if no materialization).

    Mirrors the root-input workdir injection in
    :meth:`WorkflowRuntime._populate_root_inputs`, but applies to EVERY task so
    ``ctx.workdir`` is always available (not just root tasks). Keyed on the task's
    ``TaskSnapshot.key`` so the location is stable across runs (content-addressed).
    """
    materialization = getattr(deps, "materialization", None)
    if materialization is None:
        return None
    snap = deps.snapshots.get(name)
    content_id = snap.key if snap is not None else name
    return materialization.workdir_for(content_id)


def _make_sub_runner(deps: WorkflowDeps, *, root_input: TaskInput, forward: bool):  # noqa: ANN202
    """Build the ``sub_runner`` capability for a :class:`SubWorkflow` body.

    Returns an ``async (inner_spec, config=None) -> WorkflowResult`` closure that
    runs *inner_spec* through the engine bound to this run via the PRIVATE
    ``deps.run_context`` channel — so the SubWorkflow body never touches a
    run-context itself (pure {inputs, config} contract).

    When ``forward`` is true, ``root_input`` (the outer node's resolved input —
    fan-out element, upstream output, or workspace root params) is forwarded into
    the inner spec's single entry task as its ``ctx.inputs``. When false (a bare
    root SubWorkflow with no input), the inner spec runs unchanged.

    Inner runs execute with ``persist=False``: they inherit the OUTER
    ``run_context`` (same run dir + active execution id), so letting them
    persist would rewrite ``executions/<exec_id>/workflow.json`` with the
    INNER spec's document — clobbering the parent's graph/statuses, polluting
    resume seeds, and racing under ``wf.parallel`` fan-out. The parent
    execution's document must describe the outer graph only; the SubWorkflow
    node itself is statused there by the outer run.
    """

    async def _sub_runner(inner: CompiledWorkflow, config: JSONMapping | None = None):  # noqa: ANN202
        from .runtime import WorkflowRuntime

        extra = {"root_input": root_input} if forward else {}
        return await WorkflowRuntime().execute(
            inner,
            run_context=deps.run_context,
            config=config if config is not None else deps.config,
            persist=False,
            **extra,
        )

    return _sub_runner


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
    # ``None``). The engine launches a task only after its deps are satisfied,
    # so a dep in neither set is a genuine never-ran error, not a silent ``None``.
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
