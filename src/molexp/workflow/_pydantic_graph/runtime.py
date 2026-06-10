"""WorkflowRuntime: the workflow execution facade over the structural engine.

The single concrete runtime; molexp does not abstract over runtime
backends because there is only one.  Execution modes:

- ``execute()`` — run to completion, return :class:`WorkflowResult`.
- ``start()`` — launch in background, return :class:`WorkflowExecution`.

No per-frame snapshots are written; ``workflow.json`` is opened via
:func:`.persistence.open_execution_document` for observability (per-node
:func:`.persistence.mark_task_status` updates are coalesced in memory and
flushed at bounded staleness; terminal states flush synchronously, and a
``finally``-path :func:`.persistence.close_execution_document` guarantees
the last write even when the engine raises). Resume is caller-driven via
``WorkflowResult.outputs`` + ``execute(seed_outputs=…)``.

Each ``CompiledWorkflow`` carries a frozen
:class:`~molexp.workflow._pydantic_graph.plan.ExecutionPlan` (see
:mod:`.compiler`). The runtime builds fresh state + deps per execution and
drives :func:`.engine.run_plan` — the values-on-edges scheduler; final
outputs are read from the shared, mutated ``state.results``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anyio
from mollog import get_logger

from ..materialization_store import FileMaterializationStore, MaterializationStore
from ..protocols import JSONMapping, JSONValue, RunContextLike, TaskOutput, UserDeps
from ..types import WorkflowError, WorkflowExecution, WorkflowResult
from .engine import run_plan
from .state import WorkflowDeps, WorkflowState

if TYPE_CHECKING:
    from ..cache import Caching
    from ..compiled import CompiledWorkflow, _ExperimentLike
    from .plan import ExecutionPlan


# Sentinel for ``execute(root_input=…)``: distinguishes "no forwarding" from a
# legitimately-``None`` forwarded value (a fan-out element may itself be ``None``).
_NO_ROOT_INPUT: Any = object()


def _resolve_single_root(compiled: CompiledWorkflow) -> str:
    """Return the inner spec's single entry task (the one fed a forwarded input).

    Used when a :class:`~molexp.workflow.SubWorkflow` forwards its node input into
    the inner workflow: that value becomes the entry task's ``ctx.inputs``. Prefers
    an explicit single ``entries`` declaration; otherwise computes the single
    dependency-root (a task with no upstream deps that is not a ``wf.parallel``
    body). Raises :class:`ValueError` when the entry is ambiguous, mirroring
    :meth:`SubWorkflow._resolve_output_name`.
    """
    entries = tuple(compiled._entries)
    if len(entries) == 1:
        return entries[0]
    body_names = {par.body for par in compiled._parallels}
    roots = sorted(
        reg.name for reg in compiled._tasks if not reg.depends_on and reg.name not in body_names
    )
    if len(roots) == 1:
        return roots[0]
    raise ValueError(
        f"SubWorkflow forwards an input into inner workflow {compiled.name!r}, "
        f"but it has {len(roots)} entry task(s) {roots!r}; give the inner spec a "
        f"single entry (one root task, or WorkflowCompiler(entry='<task>')) so the "
        f"forwarded input has an unambiguous destination."
    )


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
    plan: ExecutionPlan,
    state: WorkflowState,
    deps: WorkflowDeps,
) -> WorkflowState:
    """Drive a compiled :class:`ExecutionPlan` to completion and return the final state.

    The engine mutates *state* in place (each completed node records into
    ``state.results``), so the returned object is the same *state* instance
    carrying the final outputs.
    """
    await run_plan(plan, state, deps)
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


def _get_active_execution_id(run_context: RunContextLike | None) -> str | None:
    """Extract the active RunContext execution id when one is already open."""
    if run_context is None:
        return None
    getter = getattr(run_context, "_get_execution_id", None)
    if callable(getter):
        value = getter()
        if isinstance(value, str) and value:
            return value
    value = getattr(run_context, "execution_id", None)
    if isinstance(value, str) and value:
        return value
    value = getattr(run_context, "_execution_id", None)
    return value if isinstance(value, str) and value else None


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
    """Workflow runtime over the structural values-on-edges engine.

    Takes a pre-compiled :class:`~molexp.workflow.compiled.CompiledWorkflow`
    (lowered once by :meth:`WorkflowCompiler.compile`) and executes its
    ``.graph`` (an :class:`ExecutionPlan`) via :func:`.engine.run_plan`; no
    recompilation happens here. This class owns the execution facade —
    ``execute`` / ``start`` / ``run_on`` — that used to live on the
    ``Workflow`` spec object.

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
        execution_id: str | None,
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

        # Engine materialization layer: content-addressed workdir derivation +
        # task return-value persistence. Rooted at a WORKSPACE-shared
        # ``.materialize`` dir (run-independent) so a content-addressed workdir is
        # identical across runs — that is what makes cross-run reuse a byproduct
        # of content addressing and keeps root-task cache keys stable. Falls back
        # to the run dir only when the workspace root is unreachable (duck-typed
        # stub run_context); active only for a workspace run.
        materialization: MaterializationStore | None = None
        if run_context is not None:
            ws = getattr(
                getattr(getattr(run_for_deps, "experiment", None), "project", None),
                "workspace",
                None,
            )
            ws_root = getattr(ws, "root", None)
            mat_root = (
                Path(ws_root) / ".materialize"
                if ws_root is not None
                else (Path(run_dir) / ".materialize" if run_dir is not None else None)
            )
            if mat_root is not None:
                materialization = FileMaterializationStore(mat_root)

        return WorkflowDeps(
            run=run_for_deps,
            run_context=run_context,
            config=effective_config,
            user_deps=deps,
            remote_executor=None,
            run_dir=run_dir,
            execution_id=execution_id,
            registration_by_name=registration_by_name,
            parallel_decls=parallel_decls,
            loop_max_iters=loop_max_iters,
            parallel_limiters=parallel_limiters,
            cache=cache,
            snapshots=compiled.snapshots,
            materialization=materialization,
        )

    @staticmethod
    def _populate_root_inputs(
        compiled: CompiledWorkflow,
        state: WorkflowState,
        deps: WorkflowDeps,
        run_context: RunContextLike | None,
        root_input: Any = _NO_ROOT_INPUT,  # noqa: ANN401
    ) -> None:
        """Inject root-task inputs (capabilities-as-inputs + SubWorkflow forwarding).

        For each ROOT task (no upstream deps, not a ``wf.parallel`` body, not
        seeded) of a *workspace* run the engine pre-sets ``ctx.inputs = {"params":
        <run params>, "workdir": <content-addressed Path>}``. The workdir is a bare
        ``pathlib.Path`` (NEVER a navigable handle), derived from the node's
        content identity (``snapshot.key``) via the materialization layer; this
        half is a no-op without a ``run_context`` so non-workspace runs are
        unaffected.

        When ``root_input`` is provided (a :class:`SubWorkflow` forwarding its node
        input — the fan-out element, upstream output, or root params — into this
        inner spec), it is delivered to the single entry task as its ``ctx.inputs``.
        When BOTH the engine-injected ``{params, workdir}`` and the forwarded value
        are dicts, they are MERGED (forwarded keys win) so the inner entry sees the
        element AND keeps ``params`` / ``workdir``; otherwise the forwarded value
        replaces the entry input. Applies whether or not a ``run_context`` is
        present, so the inner entry sees the element even on a plain run.
        """
        body_names = {par.body for par in compiled._parallels}
        if run_context is not None:
            params = getattr(run_context, "params", None) or {}
            snapshots = compiled.snapshots
            for reg in compiled._tasks:
                name = reg.name
                if reg.depends_on or name in body_names or name in state.seeded:
                    continue
                workdir = None
                if deps.materialization is not None:
                    snap = snapshots.get(name)
                    content_id = snap.key if snap is not None else name
                    workdir = deps.materialization.workdir_for(content_id)
                state.root_inputs[name] = {"params": dict(params), "workdir": workdir}
        if root_input is not _NO_ROOT_INPUT:
            entry = _resolve_single_root(compiled)
            existing = state.root_inputs.get(entry)
            if isinstance(existing, dict) and isinstance(root_input, dict):
                # Merge: keep engine-injected params/workdir, forwarded keys win.
                state.root_inputs[entry] = {**existing, **root_input}
            else:
                state.root_inputs[entry] = root_input

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
        root_input: Any = _NO_ROOT_INPUT,  # noqa: ANN401
        persist: bool = True,
    ) -> WorkflowResult:
        """Run the workflow to completion and return a WorkflowResult.

        ``seed_outputs`` (optional) pre-populates the initial state with
        already-known task outputs; see :meth:`Workflow.execute` for the
        full contract. ``cache`` (optional) opts the run into content-
        addressed task-result caching; see :func:`_resolve_cache` for the
        precedence rules when it is omitted. ``root_input`` (optional) forwards a
        value into the spec's single entry task as its ``ctx.inputs`` — the channel
        a :class:`~molexp.workflow.SubWorkflow` uses to pass its node input (fan-out
        element / upstream output) into the inner workflow. ``persist=False``
        (engine-internal — set by the ``sub_runner`` capability for SubWorkflow
        inner runs) disables ALL ``workflow.json`` persistence for this
        execution, so a nested run inheriting the outer ``run_context`` never
        rewrites the parent execution's document: after a run containing
        SubWorkflows, ``executions/<exec_id>/workflow.json`` describes the
        OUTER graph only.
        """

        # Validate seed_outputs FAIL-FAST before any IO / scheduling work.
        state = self._build_initial_state(compiled, seed_outputs)

        resolved_run_dir = _resolve_run_dir(run_context, run_dir)
        run_id = _get_run_id(run_context)

        execution_id = (
            execution_id
            or _get_active_execution_id(run_context)
            or make_execution_id(run_id, resolved_run_dir)
        )

        persist_dir = resolved_run_dir if persist else None

        # Resume-seed integrity gate — runs BEFORE the prior workflow.json is
        # rewritten below. Seeds whose persisted snapshot key no longer matches
        # the live task code, whose persisted output is lossy, or that cannot
        # be verified (pre-upgrade document) are dropped with a warning and
        # recomputed; see ``filter_resume_seeds``. Unknown names already
        # failed fast in ``_build_initial_state`` above.
        if seed_outputs and persist_dir is not None:
            from .persistence import filter_resume_seeds

            verified = filter_resume_seeds(
                persist_dir, execution_id, seed_outputs, compiled.snapshots
            )
            if set(verified) != set(seed_outputs):
                seed_outputs = verified
                state = self._build_initial_state(compiled, seed_outputs)

        # Observability — open the execution document (initial workflow.json is
        # written synchronously under ``<run_dir>/executions/<execution_id>/``
        # so the execution-id directory always exists post-execution for
        # tooling) and register the coalescing in-memory writer: per-node
        # ``mark_task_status`` updates buffer in memory and flush at bounded
        # staleness instead of rewriting the document per transition. The
        # graph runner persists no per-frame snapshots (resume is caller-driven
        # via seed_outputs).
        if persist_dir is not None:
            from .persistence import open_execution_document

            open_execution_document(persist_dir, execution_id, compiled=compiled)

        try:
            workflow_deps = self._build_deps(
                compiled,
                run_context=run_context,
                run_dir=resolved_run_dir,
                # ``deps.execution_id`` gates per-task workflow.json status
                # writes (``mark_task_status``); a persistence-off (nested)
                # run must not touch the parent's document.
                execution_id=execution_id if persist else None,
                config=config,
                deps=deps,
                cache=_resolve_cache(cache, self.cache, run_context),
            )

            self._populate_root_inputs(compiled, state, workflow_deps, run_context, root_input)

            result_state: WorkflowState = await _run_compiled(compiled.graph, state, workflow_deps)

            # Propagate task-failure to the workspace's RunContext so it can
            # tag the final run.status as failed when the caller's
            # ``with run.start() as ctx: workflow.execute(run_context=ctx)``
            # block exits cleanly. Without this back-channel the failure
            # only surfaces in WorkflowResult.status — which the CLI does
            # not consult — and run.status defaults to ``succeeded``.
            if result_state.failed and run_context is not None:
                _record_run_failure(run_context, result_state.error)
            if persist_dir is not None:
                from .persistence import mark_workflow_finished

                mark_workflow_finished(
                    persist_dir,
                    execution_id,
                    status="failed" if result_state.failed else "completed",
                    outputs=result_state.results,
                    error=result_state.error,
                )

            return WorkflowResult(
                status="failed" if result_state.failed else "completed",
                outputs=result_state.results,
                run_id=run_id,
                execution_id=execution_id,
            )
        except WorkflowError as exc:
            # Programming errors in the workflow definition / task body
            # (CycleError, UnknownRouteError, MissingRouteError, …)
            # propagate to the caller.
            if persist_dir is not None:
                from .persistence import mark_workflow_finished

                mark_workflow_finished(
                    persist_dir,
                    execution_id,
                    status="failed",
                    outputs=dict(state.results),
                    error=str(exc),
                )
            raise
        except Exception as exc:
            logger.exception(f"Workflow {compiled.name!r} execution failed")
            if run_context is not None:
                _record_run_failure(run_context, str(exc))
            if persist_dir is not None:
                from .persistence import mark_workflow_finished

                mark_workflow_finished(
                    persist_dir,
                    execution_id,
                    status="failed",
                    outputs=dict(state.results),
                    error=str(exc),
                )
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
        finally:
            # Guarantee the last in-memory document state lands on disk even
            # when the engine raises something the arms above never see
            # (BaseException / cancellation): flush coalesced-but-unwritten
            # node records and end the writer lifecycle. No-op on the normal
            # paths (mark_workflow_finished already flushed + closed) and for
            # persistence-off (SubWorkflow inner) executions.
            from .persistence import close_execution_document

            close_execution_document(persist_dir, execution_id)

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
        execution_id = (
            execution_id
            or _get_active_execution_id(run_context)
            or make_execution_id(run_id, resolved_run_dir)
        )

        # Observability — see ``execute()`` for the rationale (initial write +
        # coalescing in-memory writer; closed in ``_bg``'s ``finally``).
        if resolved_run_dir is not None:
            from .persistence import open_execution_document

            open_execution_document(resolved_run_dir, execution_id, compiled=compiled)

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
                    execution_id=execution_id,
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
                if resolved_run_dir is not None:
                    from .persistence import mark_workflow_finished

                    mark_workflow_finished(
                        resolved_run_dir,
                        execution_id,
                        status="failed" if result_state.failed else "completed",
                        outputs=result_state.results,
                        error=result_state.error,
                    )
            except Exception as exc:
                # ``seed_state`` is mutated in place by the graph runner — it
                # carries the results of every task that completed before the
                # raise. Preserve them for ``seed_outputs=`` resume.
                handle._result = WorkflowResult(
                    status="failed",
                    outputs=dict(seed_state.results),
                    run_id=handle.run_id,
                    execution_id=execution_id,
                )
                if resolved_run_dir is not None:
                    from .persistence import mark_workflow_finished

                    mark_workflow_finished(
                        resolved_run_dir,
                        execution_id,
                        status="failed",
                        outputs=dict(seed_state.results),
                        error=str(exc),
                    )
                logger.exception(f"Background workflow {compiled.name!r} failed")
            finally:
                # Terminal-flush guarantee for paths the except-arm never sees
                # (cancellation): land the last document state, end the writer
                # lifecycle. No-op when mark_workflow_finished already closed.
                from .persistence import close_execution_document

                close_execution_document(resolved_run_dir, execution_id)
                handle._done_event.set()

        handle._task = asyncio.create_task(_bg())
        return handle

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
        run = cast("Any", experiment).add_run(params=params_dict)
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
