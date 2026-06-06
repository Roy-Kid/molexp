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

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from mollog import get_logger
from pydantic_graph import End

from ..context import TaskContext
from ..protocols import (
    AssetsViewLike,
    JSONMapping,
    JSONValue,
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
    MissingRouteError,
    Next,
    OutEdges,
    UnknownTaskError,
)
from .state import WorkflowDeps, WorkflowState

if TYPE_CHECKING:
    from .._graph_decl import TaskRegistration

logger = get_logger(__name__)


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


def _is_json_safe(value: object) -> bool:
    """Return True iff *value* round-trips through ``json.dumps`` cleanly."""
    import json

    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _cache_inputs(inputs: TaskInput) -> dict[str, JSONValue]:
    """Wrap the collected upstream inputs as the cache ``inputs`` mapping."""
    return {"inputs": inputs}


def _artifact_manifest(deps: WorkflowDeps, name: str) -> list[dict[str, JSONValue]]:
    """Build the JSON artifact manifest for task *name* in the current run.

    Queries the current run's catalog view for artifacts whose producer
    task is *name* and snapshots each as a JSON dict
    ``{name, kind, content_hash, asset_id}``. Returns ``[]`` when no
    workspace asset view is reachable.
    """
    run_context = deps.run_context
    if run_context is None:
        return []
    # The scope-filtered asset view lives on the Run (``run_context.run``);
    # fall back to a direct ``.assets`` on the context for duck-typed stubs.
    run = getattr(run_context, "run", None)
    assets_view = getattr(run, "assets", None) or getattr(run_context, "assets", None)
    query = getattr(assets_view, "query", None)
    if not callable(query):
        return []
    try:
        found = query(producer_task=name, kind="artifact")
    except Exception:
        return []
    manifest: list[dict[str, JSONValue]] = []
    for asset in found or []:
        content_hash = getattr(asset, "content_hash", None)
        if not content_hash:
            continue
        manifest.append(
            {
                "name": getattr(asset, "name", None),
                "kind": getattr(asset, "kind", "artifact"),
                "content_hash": content_hash,
                "asset_id": getattr(asset, "asset_id", None),
            }
        )
    return manifest


def _reregister_artifacts(deps: WorkflowDeps, name: str, manifest: list[dict]) -> None:
    """Re-register cached artifacts into the current run by content-hash.

    Idempotent catalog upsert keyed on ``(kind, content_hash)`` pointing at
    bytes already present in the content-addressed store — no recompute, no
    byte recopy. Entries whose bytes are absent (fresh workspace) are
    skipped gracefully. Reached purely through duck-typed ``run_context``
    surface so the workflow layer keeps its decoupling from a concrete
    workspace import.
    """
    run_context = deps.run_context
    if run_context is None or not manifest:
        return
    run = getattr(run_context, "run", None)
    scope = getattr(run, "scope", None)
    # Reach the workspace catalog through the run's ancestry (duck-typed).
    experiment = getattr(run, "experiment", None)
    project = getattr(experiment, "project", None)
    workspace = getattr(project, "workspace", None)
    catalog = getattr(workspace, "catalog", None)
    reregister = getattr(catalog, "reregister_artifact", None)
    if scope is None or not callable(reregister):
        return
    for entry in manifest:
        content_hash = entry.get("content_hash")
        if not content_hash:
            continue
        try:
            reregister(
                name=entry.get("name"),
                content_hash=content_hash,
                target_scope=scope,
                producer_task=name,
            )
        except Exception:
            logger.debug(f"cache: re-register of artifact {entry.get('name')!r} skipped")


async def run_task_body_cached(
    name: str,
    deps: WorkflowDeps,
    state: WorkflowState,
) -> TaskOutput:
    """Run task *name*'s body with content-addressed result caching.

    Gating (caller pre-checks ``deps.cache is not None``, non-actor task,
    ``name in deps.snapshots``):

    * collect the upstream inputs once and wrap them JSON-safely as the
      cache ``inputs`` payload;
    * ``cache.get`` → on HIT, re-register the cached artifact manifest into
      the current run and return the recorded ``result`` WITHOUT running the
      body (the per-task body counter must not increment);
    * on MISS, run the body, assemble the produced-artifact manifest, and
      ``cache.put({"result": raw, "artifacts": manifest})``. Non-JSON-safe
      inputs / results degrade gracefully — the body still runs and the put
      is skipped.

    The returned raw value is routed by the caller through the SAME
    ``_classify_return`` path as a plain return, so branch / End semantics
    hold identically on hits and misses.
    """
    registration = deps.registration_by_name.get(name)
    if registration is None:
        raise UnknownTaskError(f"run_task_body_cached: unknown task {name!r}")

    cache = deps.cache
    snapshot = deps.snapshots.get(name)
    assert cache is not None and snapshot is not None  # caller-gated

    inputs = _collect_upstream_outputs(registration, state)
    cacheable = _is_json_safe(inputs)
    cache_inputs = _cache_inputs(inputs)

    if cacheable:
        try:
            payload = cache.get(snapshot, cache_inputs)
        except Exception:
            payload = None
        if payload is not None:
            artifacts = payload.get("artifacts", [])
            if isinstance(artifacts, list):
                _reregister_artifacts(deps, name, [a for a in artifacts if isinstance(a, dict)])
            return payload.get("result")

    raw = await run_task_body(name, deps, state)

    if cacheable and _is_json_safe(raw):
        manifest = _artifact_manifest(deps, name)
        result_payload = cast("dict[str, JSONValue]", {"result": raw, "artifacts": manifest})
        try:
            cache.put(snapshot, cache_inputs, result_payload)
        except Exception:
            logger.debug(f"cache: put for task {name!r} skipped (non-serializable)")
    return raw


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
    "_EndTok",
    "_Failure",
    "_Trigger",
    "_classify_return",
    "run_task_body",
    "run_task_body_cached",
]
