"""WorkflowRuntime: pydantic-graph-backed workflow runtime.

The single concrete runtime; molexp does not abstract over runtime
backends because there is only one.  Execution modes:

- ``execute()`` — run to completion, return :class:`WorkflowResult`.
- ``start()`` — launch in background, return :class:`WorkflowExecution`.
- ``iter()`` / ``stream()`` — async step-by-step iteration.

``resume()`` is removed — the new ``GraphBuilder``-based API does not
expose ``iter_from_persistence``.  No per-frame snapshots are written; only
an initial ``workflow.json`` is emitted via
:func:`.persistence.write_initial_workflow_json` for observability. Resume is
caller-driven via ``WorkflowResult.outputs`` + ``execute(seed_outputs=…)``.

Each ``CompiledWorkflow`` carries a genuine ``pydantic_graph`` ``Graph``
(one Step per task; see :mod:`.compiler`). The runtime builds fresh state
+ deps per execution and drives ``compiled.graph.run(state=…, deps=…,
inputs=None)``; final outputs are read from the shared, mutated
``state.results``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
from mollog import get_logger

from ..protocols import JSONMapping, JSONValue, RunContextLike, TaskOutput, UserDeps
from ..types import WorkflowError, WorkflowExecution, WorkflowResult
from .state import WorkflowDeps, WorkflowState

if TYPE_CHECKING:
    from pydantic_graph.graph_builder import Graph

    from ..cache import Caching
    from ..compiled import CompiledWorkflow, _ExperimentLike


def _resolve_cache(
    explicit: Caching | None,
    instance_cache: Caching | None,
    run_context: RunContextLike | None,
) -> Caching | None:
    """Pick the effective :class:`Caching` for one execution.

    Resolution order (spec workflow-refactor-04 §Plumbing):

    1. an explicit ``cache=`` kwarg passed to ``execute`` / ``start`` / …;
    2. the runtime's flat ``self.cache`` instance attribute;
    3. auto-derived from a workspace ``run_context`` that exposes a cache
       store (``run_context.run.experiment.project.workspace.cache`` →
       ``.as_cache_store()``), probed defensively via ``getattr`` so a
       duck-typed / stub context never raises;
    4. ``None`` (caching off — identical behaviour to before this spec).
    """
    if explicit is not None:
        return explicit
    if instance_cache is not None:
        return instance_cache
    return _auto_cache_from_run_context(run_context)


def _auto_cache_from_run_context(run_context: RunContextLike | None) -> Caching | None:
    """Best-effort: build a workspace-backed ``Caching`` from a run_context.

    Returns ``None`` whenever the duck-typed surface does not expose a
    workspace cache store — a plain stub run_context, a run_context with
    no workspace ancestry, etc. Never raises.
    """
    if run_context is None:
        return None
    run = getattr(run_context, "run", None)
    if run is None:
        return None
    experiment = getattr(run, "experiment", None)
    project = getattr(experiment, "project", None)
    workspace = getattr(project, "workspace", None)
    cache_folder = getattr(workspace, "cache", None)
    as_cache_store = getattr(cache_folder, "as_cache_store", None)
    if not callable(as_cache_store):
        return None
    try:
        store = as_cache_store()
    except Exception:
        return None
    from ..cache import Caching

    return Caching(store=store)


logger = get_logger(__name__)


async def _run_compiled(
    graph: Graph[WorkflowState, WorkflowDeps, None, None],
    state: WorkflowState,
    deps: WorkflowDeps,
) -> WorkflowState:
    """Drive a compiled pg ``Graph`` to completion and return the final state.

    The graph mutates *state* in place (molexp tasks read/write
    ``state.results`` directly), so the returned object is the same *state*
    instance carrying the final outputs. pydantic-graph 1.x requires an
    explicit ``inputs=`` start token; the lowered entry Steps ignore it
    (``None``) and read their upstreams from ``state.results``.
    """
    await graph.run(state=state, deps=deps, inputs=None)
    return state


def _resolve_run_dir(
    run_context: RunContextLike | None, explicit_run_dir: str | Path | None
) -> Path | None:
    """Pick the run directory: explicit ``run_dir=`` wins, else duck-type
    ``run_context.work_dir``."""
    if explicit_run_dir is not None:
        return Path(explicit_run_dir)
    if run_context is None:
        return None
    work_dir = getattr(run_context, "work_dir", None)
    if work_dir is not None:
        return Path(work_dir)
    return None


def _get_run_id(run_context: RunContextLike | None) -> str | None:
    """Extract a stable run identifier from a duck-typed run_context."""
    if run_context is None:
        return None
    run = getattr(run_context, "run", None)
    if run is None:
        return None
    return getattr(run, "id", getattr(run, "run_id", None))


def _record_run_failure(run_context: RunContextLike | None, error: str | None) -> None:
    """Mark task-failure on a duck-typed run_context via its typed
    :meth:`RunContextLike.mark_failed`.

    A task body can fail without the exception propagating out of ``execute()``
    (e.g. a ``wf.parallel`` element capturing its error). The workspace's
    ``RunContext`` resolves an exception-free ``with ctx:`` exit to a failed
    run-status by consulting what ``mark_failed`` records, so the CLI surfaces
    the failure even though no exception reached it.
    """
    mark_failed = getattr(run_context, "mark_failed", None)
    if callable(mark_failed):
        mark_failed(error)


def make_execution_id(run_id: str | None, run_dir: Path | None) -> str:
    """Build a human-readable execution ID.

    First execution: ``exec-{run_id}``
    Retries:         ``exec-{run_id}-2``, ``exec-{run_id}-3``, …

    Falls back to a human-readable random name (e.g. ``exec-serene-mixing-reddy``)
    when *run_id* is unavailable.

    Spec 04 §6 — promoted to the public API. Re-exported as
    :func:`molexp.workflow.make_execution_id`. ``submit_molq`` plugins
    must use the public name; reaching into ``_pydantic_graph`` for
    this helper is rejected by ``test_submit_molq_plugins_do_not_reach_into_pydantic_graph``.
    """
    from molexp.workflow._names import generate_name
    from molexp.workspace.utils import derive_execution_id

    if run_id is None:
        return f"exec-{generate_name()}"
    if run_dir is None:
        return f"exec-{run_id}"

    return derive_execution_id(run_id, Path(run_dir) / "executions")


class WorkflowRuntime:
    """Workflow runtime powered by pydantic-graph.

    Takes a pre-compiled :class:`~molexp.workflow.compiled.CompiledWorkflow`
    (lowered once by :meth:`WorkflowCompiler.compile`) and executes its
    ``.graph``; no recompilation happens here. This class owns the
    execution facade — ``execute`` / ``start`` / ``iter`` / ``stream`` /
    ``run_on`` — that used to live on the ``Workflow`` spec object.

    ``self.cache`` is a flat, settable :class:`~molexp.workflow.cache.Caching`
    instance attribute (default ``None`` — caching off). It is the lowest
    priority cache source; an explicit ``cache=`` kwarg on any execution
    method wins, and a workspace-backed cache is auto-derived from a
    workspace ``run_context`` when neither is set (see :func:`_resolve_cache`).
    """

    def __init__(self) -> None:
        self.cache: Caching | None = None

    @staticmethod
    def _build_initial_state(
        compiled: CompiledWorkflow,
        seed_outputs: Mapping[str, TaskOutput] | None,
    ) -> WorkflowState:
        """Construct the initial :class:`WorkflowState`, optionally seeded.

        When ``seed_outputs`` is non-empty:

        * Every key is validated against the spec's registered task names
          so unknown names fail fast (the ``planmode-review-repair-loop``
          ``ac-006`` contract).
        * Seeded names land in ``state.results`` + ``state.completed`` (so
          downstream tasks find their values) and ``state.seeded`` (so the
          seeded task's own Step skips invoking the body while still
          routing normally through the lowered graph).
        """
        if not seed_outputs:
            return WorkflowState()
        registered = {t.name for t in compiled._tasks}
        unknown = sorted(set(seed_outputs) - registered)
        if unknown:
            raise ValueError(
                f"execute(seed_outputs=...): unknown task name(s) "
                f"{unknown!r}; registered tasks: {sorted(registered)}"
            )
        return WorkflowState.from_seed(seed_outputs)

    @staticmethod
    def _build_deps(
        compiled: CompiledWorkflow,
        *,
        run_context: RunContextLike | None,
        run_dir: Path | None,
        config: JSONMapping | None,
        deps: UserDeps,
        cache: Caching | None = None,
    ) -> WorkflowDeps:
        """Build a fresh :class:`WorkflowDeps` for one execution.

        Topology fields (registration_by_name / parallel_decls /
        loop_max_iters) are derived from the compiled artifact; one fresh
        :class:`anyio.CapacityLimiter` is built per ``wf.parallel`` body,
        sized to its ``max_concurrency``. When *run_context* is provided
        its attached ``.config`` takes precedence over the *config* kwarg.

        ``cache`` (the resolved effective :class:`Caching`, or ``None``) and
        the compiled artifact's per-task ``snapshots`` are threaded onto the
        deps so the per-task Step cache hook can derive a cache key and
        get / put results.
        """
        if run_context is not None:
            ctx_config = getattr(run_context, "config", None)
            effective_config = ctx_config if ctx_config is not None else config
        else:
            effective_config = config

        run_for_deps = getattr(run_context, "run", None) if run_context is not None else None

        # Static topology maps are derived once and cached on the (frozen)
        # compiled artifact — reused across every execution. Only the capacity
        # limiters must be fresh per run (live anyio objects).
        registration_by_name = compiled.registration_by_name
        parallel_decls = compiled.parallel_decls_by_body
        loop_max_iters = compiled.loop_max_iters
        parallel_limiters = {
            par.body: anyio.CapacityLimiter(par.max_concurrency) for par in compiled._parallels
        }

        return WorkflowDeps(
            run=run_for_deps,
            run_context=run_context,
            config=effective_config,
            user_deps=deps,
            remote_executor=None,
            run_dir=run_dir,
            registration_by_name=registration_by_name,
            parallel_decls=parallel_decls,
            loop_max_iters=loop_max_iters,
            parallel_limiters=parallel_limiters,
            cache=cache,
            snapshots=compiled.snapshots,
        )

    # ── execute ──────────────────────────────────────────────────────────────

    async def execute(
        self,
        compiled: CompiledWorkflow,
        *,
        run_context: RunContextLike | None = None,
        run_dir: str | Path | None = None,
        config: JSONMapping | None = None,
        deps: UserDeps = None,
        execution_id: str | None = None,
        seed_outputs: Mapping[str, TaskOutput] | None = None,
        cache: Caching | None = None,
    ) -> WorkflowResult:
        """Run the workflow to completion and return a WorkflowResult.

        ``seed_outputs`` (optional) pre-populates the initial state with
        already-known task outputs; see :meth:`Workflow.execute` for the
        full contract. ``cache`` (optional) opts the run into content-
        addressed task-result caching; see :func:`_resolve_cache` for the
        precedence rules when it is omitted.
        """

        # Validate seed_outputs FAIL-FAST before any IO / scheduling work.
        state = self._build_initial_state(compiled, seed_outputs)

        resolved_run_dir = _resolve_run_dir(run_context, run_dir)
        run_id = _get_run_id(run_context)

        execution_id = execution_id or make_execution_id(run_id, resolved_run_dir)

        # Observability — write the initial workflow.json under
        # ``<run_dir>/executions/<execution_id>/`` so the execution-id directory
        # always exists post-execution for tooling. The graph runner persists no
        # per-frame snapshots (resume is caller-driven via seed_outputs); this
        # one-shot write is all that happens.
        if resolved_run_dir is not None:
            from .persistence import write_initial_workflow_json

            write_initial_workflow_json(resolved_run_dir, execution_id)

        try:
            workflow_deps = self._build_deps(
                compiled,
                run_context=run_context,
                run_dir=resolved_run_dir,
                config=config,
                deps=deps,
                cache=_resolve_cache(cache, self.cache, run_context),
            )

            result_state: WorkflowState = await _run_compiled(compiled.graph, state, workflow_deps)

            # Propagate task-failure to the workspace's RunContext so it can
            # tag the final run.status as failed when the caller's
            # ``with run.start() as ctx: workflow.execute(run_context=ctx)``
            # block exits cleanly. Without this back-channel the failure
            # only surfaces in WorkflowResult.status — which the CLI does
            # not consult — and run.status defaults to ``succeeded``.
            if result_state.failed and run_context is not None:
                _record_run_failure(run_context, result_state.error)

            return WorkflowResult(
                status="failed" if result_state.failed else "completed",
                outputs=result_state.results,
                run_id=run_id,
                execution_id=execution_id,
            )
        except WorkflowError:
            # Programming errors in the workflow definition / task body
            # (CycleError, UnknownRouteError, MissingRouteError, …)
            # propagate to the caller.
            raise
        except Exception as exc:
            logger.exception(f"Workflow {compiled.name!r} execution failed")
            if run_context is not None:
                _record_run_failure(run_context, str(exc))
            # ``state`` is mutated in place by the graph runner, so it still
            # holds every task result recorded before the raise. Preserve them
            # so the caller can resume via ``seed_outputs=`` instead of
            # recomputing completed (often expensive) tasks.
            return WorkflowResult(
                status="failed",
                outputs=dict(state.results),
                run_id=run_id,
                execution_id=execution_id,
            )

    # ── start ────────────────────────────────────────────────────────────────

    async def start(
        self,
        compiled: CompiledWorkflow,
        *,
        run_context: RunContextLike | None = None,
        run_dir: str | Path | None = None,
        config: JSONMapping | None = None,
        deps: UserDeps = None,
        execution_id: str | None = None,
        seed_outputs: Mapping[str, TaskOutput] | None = None,
        cache: Caching | None = None,
    ) -> WorkflowExecution:
        """Launch workflow as background asyncio task.

        See :meth:`execute` for ``seed_outputs`` semantics; the same
        fail-fast validation applies before scheduling the background
        task.
        """
        # Fail-fast on bad seeds so the caller observes the ValueError
        # synchronously, not via an awaited handle.
        seed_state = self._build_initial_state(compiled, seed_outputs)
        resolved_run_dir = _resolve_run_dir(run_context, run_dir)
        run_id = _get_run_id(run_context)
        execution_id = execution_id or make_execution_id(run_id, resolved_run_dir)

        # Observability — see ``execute()`` for the rationale.
        if resolved_run_dir is not None:
            from .persistence import write_initial_workflow_json

            write_initial_workflow_json(resolved_run_dir, execution_id)

        handle = _GraphWorkflowExecution(
            execution_id=execution_id,
            workflow_id=compiled.workflow_id,
            run_id=run_id,
        )

        resolved_cache = _resolve_cache(cache, self.cache, run_context)

        async def _bg() -> None:
            try:
                workflow_deps = self._build_deps(
                    compiled,
                    run_context=run_context,
                    run_dir=resolved_run_dir,
                    config=config,
                    deps=deps,
                    cache=resolved_cache,
                )
                result_state: WorkflowState = await _run_compiled(
                    compiled.graph, seed_state, workflow_deps
                )
                handle._result = WorkflowResult(
                    status="failed" if result_state.failed else "completed",
                    outputs=result_state.results,
                    run_id=run_id,
                    execution_id=execution_id,
                )
            except Exception:
                # ``seed_state`` is mutated in place by the graph runner — it
                # carries the results of every task that completed before the
                # raise. Preserve them for ``seed_outputs=`` resume.
                handle._result = WorkflowResult(
                    status="failed",
                    outputs=dict(seed_state.results),
                    run_id=handle.run_id,
                    execution_id=execution_id,
                )
                logger.exception(f"Background workflow {compiled.name!r} failed")
            finally:
                handle._done_event.set()

        handle._task = asyncio.create_task(_bg())
        return handle

    # ── iter ─────────────────────────────────────────────────────────────────

    def iter(
        self,
        compiled: CompiledWorkflow,
        *,
        run_context: RunContextLike | None = None,
        run_dir: str | Path | None = None,
        config: JSONMapping | None = None,
        deps: UserDeps = None,
        seed_outputs: Mapping[str, TaskOutput] | None = None,
        cache: Caching | None = None,
    ) -> Any:  # noqa: ANN401
        """Return an async context manager for step-by-step iteration.

        See :meth:`execute` for ``seed_outputs`` / ``cache`` semantics.
        """
        state = self._build_initial_state(compiled, seed_outputs)
        resolved_run_dir = _resolve_run_dir(run_context, run_dir)
        workflow_deps = self._build_deps(
            compiled,
            run_context=run_context,
            run_dir=resolved_run_dir,
            config=config,
            deps=deps,
            cache=_resolve_cache(cache, self.cache, run_context),
        )
        return compiled.graph.iter(state=state, deps=workflow_deps, inputs=None)

    # ── stream ───────────────────────────────────────────────────────────────

    def stream(
        self,
        compiled: CompiledWorkflow,
        *,
        run_context: RunContextLike | None = None,
        run_dir: str | Path | None = None,
        config: JSONMapping | None = None,
        deps: UserDeps = None,
        seed_outputs: Mapping[str, TaskOutput] | None = None,
        cache: Caching | None = None,
    ) -> Any:  # noqa: ANN401
        """Alias for iter() — streaming Actor support in a future phase."""
        return self.iter(
            compiled,
            run_context=run_context,
            run_dir=run_dir,
            config=config,
            deps=deps,
            seed_outputs=seed_outputs,
            cache=cache,
        )

    # ── run_on ─────────────────────────────────────────────────────────────────

    async def run_on(
        self,
        compiled: CompiledWorkflow,
        experiment: _ExperimentLike,
        *,
        parameters: Mapping[str, JSONValue] | None = None,
        deps: UserDeps = None,
        profile_config: object | None = None,
        config: JSONMapping | None = None,
        cache: Caching | None = None,
    ) -> WorkflowResult:
        """Build a fresh Run on *experiment*, execute *compiled*, return the result.

        Relocated from the old ``Workflow.run_on``. Does NOT bind the
        workflow to the experiment; register it explicitly via a
        :class:`~molexp.workflow.binding.WorkflowBindingRegistry` (or pass
        ``experiment=`` to :meth:`WorkflowCompiler.compile`) if you need it
        recoverable after process restart.
        """
        params_dict = dict(parameters) if parameters is not None else None
        run = experiment.add_run(parameters=params_dict)  # type: ignore[attr-defined]
        with run.start(profile_config=profile_config) as run_ctx:
            result = await self.execute(
                compiled, run_context=run_ctx, config=config, deps=deps, cache=cache
            )
        if result.status != "completed":
            err = run.metadata.error
            err_msg = (
                f"workflow {compiled.name!r} ended with status {result.status!r}: "
                f"{err.type}: {err.message}"
                if err is not None
                else f"workflow {compiled.name!r} ended with status {result.status!r}"
            )
            raise RuntimeError(err_msg)
        return result


class _GraphWorkflowExecution(WorkflowExecution):
    """Concrete WorkflowExecution returned by start()."""

    def __init__(self, execution_id: str, workflow_id: str, run_id: str | None) -> None:
        super().__init__(
            execution_id=execution_id,
            workflow_id=workflow_id,
            run_id=run_id,
        )
        self._result: WorkflowResult | None = None
        self._done_event: asyncio.Event = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def wait(self) -> WorkflowResult:
        """Block until the workflow finishes and return the result."""
        await self._done_event.wait()
        assert self._result is not None
        return self._result

    async def cancel(self) -> None:
        """Cancel the background task."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._done_event.set()
