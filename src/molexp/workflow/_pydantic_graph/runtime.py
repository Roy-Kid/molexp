"""GraphWorkflowRuntime: pydantic-graph-backed workflow runtime.

The single concrete runtime; molexp does not abstract over runtime
backends because there is only one.  Execution modes:

- ``execute()`` — run to completion, return :class:`WorkflowResult`.
- ``start()`` — launch in background, return :class:`WorkflowExecution`.
- ``iter()`` / ``stream()`` — async step-by-step iteration.

``resume()`` is removed — the new ``GraphBuilder``-based API does not
expose ``iter_from_persistence``.  Per-frame snapshots are still written
by :class:`RunStorePersistence` for observability, but they are no longer
injected into the graph runner.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mollog import get_logger
from pydantic_graph.graph_builder import GraphBuilder

from ..protocols import JSONMapping, RunContextLike, TaskOutput, UserDeps
from ..types import WorkflowError, WorkflowExecution, WorkflowResult
from .compiler import CompiledWorkflow, WorkflowGraphCompiler
from .node import WorkflowStep
from .state import WorkflowDeps, WorkflowState

if TYPE_CHECKING:
    from ..spec import Workflow

logger = get_logger(__name__)

_compiler = WorkflowGraphCompiler()

# Single-track: ``WorkflowStep`` is the only pg BaseNode molexp exposes.
# GraphBuilder auto-detects edges from WorkflowStep's return type hint
# (``WorkflowStep | End[WorkflowState]``).
_builder = GraphBuilder[WorkflowState, WorkflowDeps, None, WorkflowState](
    name="workflow",
    state_type=WorkflowState,
    deps_type=WorkflowDeps,
    output_type=WorkflowState,
)
_ws_edge = _builder.node(WorkflowStep)
_builder.add(_ws_edge)
_builder.add_edge(_builder.start_node, _ws_edge.sources[0])
_GRAPH = _builder.build()


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
    """Mark task-failure on a duck-typed run_context's ``_context.status`` dict.

    The workspace's :class:`molexp.workspace.run.RunContext` reads
    ``self._context.status.get("run")`` from ``__exit__`` to decide whether
    the run finished as ``succeeded`` or ``failed``. The runtime writes to
    that dict here so the CLI's exception-free ``with ctx:`` exit still
    surfaces task-body failures as a failed run-status.
    """
    inner = getattr(run_context, "_context", None)
    if inner is None:
        return
    status = getattr(inner, "status", None)
    if status is None or not hasattr(status, "__setitem__"):
        return
    status["run"] = "failed"
    if error and hasattr(inner, "errors") and isinstance(inner.errors, dict):
        inner.errors.setdefault("run", {"message": error})


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

    if run_id is None:
        return f"exec-{generate_name()}"

    base = f"exec-{run_id}"
    if run_dir is None:
        return base

    exec_root = run_dir / "executions"
    if not exec_root.exists():
        return base

    existing = [p for p in exec_root.iterdir() if p.name.startswith(base)]
    if not existing:
        return base
    return f"{base}-{len(existing) + 1}"


class GraphWorkflowRuntime:
    """Workflow runtime powered by pydantic-graph.

    Caches compiled graphs by workflow_id so repeated executions of the
    same spec skip recompilation.
    """

    def __init__(self) -> None:
        self._compiled: dict[str, CompiledWorkflow] = {}

    def _get_compiled(self, spec: Workflow) -> CompiledWorkflow:
        wf_id = spec.workflow_id
        if wf_id not in self._compiled:
            self._compiled[wf_id] = _compiler.compile(spec)
        return self._compiled[wf_id]

    @staticmethod
    def _build_initial_state(
        spec: Workflow,
        seed_outputs: Mapping[str, TaskOutput] | None,
    ) -> WorkflowState:
        """Construct the initial :class:`WorkflowState`, optionally seeded.

        When ``seed_outputs`` is non-empty:

        * Every key is validated against the spec's registered task names
          so unknown names fail fast (the ``planmode-review-repair-loop``
          ``ac-006`` contract).
        * Seeded names land in ``state.completed`` (so the data-dep
          satisfaction check passes for tasks that depend on them) and
          ``state.seeded`` (so the frontier filter skips invoking the
          seeded task's body).
        * The initial ``state.pending_targets`` is computed by forward-
          walking the spec's ``depends_on`` graph from each seeded name
          and collecting non-seeded children. This bypasses the spec's
          ``entry_frontier`` so that workflows where the seeded set
          covers the entry frontier still advance to downstream nodes.
        """
        if not seed_outputs:
            return WorkflowState()
        registered = {t.name for t in spec._tasks}
        unknown = sorted(set(seed_outputs) - registered)
        if unknown:
            raise ValueError(
                f"Workflow.execute(seed_outputs=...): unknown task name(s) "
                f"{unknown!r}; registered tasks: {sorted(registered)}"
            )

        # Build a forward map: name → list of children (downstream tasks).
        forward: dict[str, list[str]] = {}
        for t in spec._tasks:
            for dep in t.depends_on:
                forward.setdefault(dep, []).append(t.name)

        seeded_set = set(seed_outputs)
        pending: list[str] = []
        seen: set[str] = set()
        for seed_name in seed_outputs:
            for child in forward.get(seed_name, ()):
                if child in seeded_set or child in seen:
                    continue
                seen.add(child)
                pending.append(child)

        state = WorkflowState.from_seed(seed_outputs)
        if pending:
            state = state.set_pending(pending)
        return state

    @staticmethod
    def _build_deps(
        compiled: CompiledWorkflow,
        *,
        run_context: RunContextLike | None,
        run_dir: Path | None,
        config: JSONMapping | None,
        deps: UserDeps,
    ) -> WorkflowDeps:
        """Build WorkflowDeps from the duck-typed run_context payload.

        When *run_context* is provided, its attached ``.config`` takes
        precedence over the explicit *config* kwarg.
        """
        if run_context is not None:
            ctx_config = getattr(run_context, "config", None)
            effective_config = ctx_config if ctx_config is not None else config
        else:
            effective_config = config

        run_for_deps = getattr(run_context, "run", None) if run_context is not None else None

        return compiled.make_deps(
            run=run_for_deps,
            run_context=run_context,
            config=effective_config,
            user_deps=deps,
            remote_executor=None,
            run_dir=run_dir,
        )

    # ── execute ──────────────────────────────────────────────────────────────

    async def execute(
        self,
        spec: Workflow,
        *,
        run_context: RunContextLike | None = None,
        run_dir: str | Path | None = None,
        config: JSONMapping | None = None,
        deps: UserDeps = None,
        execution_id: str | None = None,
        seed_outputs: Mapping[str, TaskOutput] | None = None,
    ) -> WorkflowResult:
        """Run the workflow to completion and return a WorkflowResult.

        ``seed_outputs`` (optional) pre-populates the initial state with
        already-known task outputs; see :meth:`Workflow.execute` for the
        full contract.
        """
        compiled = self._get_compiled(spec)

        # Validate seed_outputs FAIL-FAST before any IO / scheduling work.
        state = self._build_initial_state(spec, seed_outputs)

        resolved_run_dir = _resolve_run_dir(run_context, run_dir)
        run_id = _get_run_id(run_context)

        execution_id = execution_id or make_execution_id(run_id, resolved_run_dir)

        try:
            workflow_deps = self._build_deps(
                compiled,
                run_context=run_context,
                run_dir=resolved_run_dir,
                config=config,
                deps=deps,
            )

            result_state: WorkflowState = await _GRAPH.run(state=state, deps=workflow_deps)

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
            # (CycleError, UnknownRouteError, MissingRouteError,
            # WorkflowDeadlockError, …) propagate to the caller.
            raise
        except Exception as exc:
            logger.exception(f"Workflow {spec.name!r} execution failed")
            if run_context is not None:
                _record_run_failure(run_context, str(exc))
            return WorkflowResult(
                status="failed",
                outputs={},
                run_id=run_id,
                execution_id=execution_id,
            )

    # ── start ────────────────────────────────────────────────────────────────

    async def start(
        self,
        spec: Workflow,
        *,
        run_context: RunContextLike | None = None,
        run_dir: str | Path | None = None,
        config: JSONMapping | None = None,
        deps: UserDeps = None,
        execution_id: str | None = None,
        seed_outputs: Mapping[str, TaskOutput] | None = None,
    ) -> WorkflowExecution:
        """Launch workflow as background asyncio task.

        See :meth:`execute` for ``seed_outputs`` semantics; the same
        fail-fast validation applies before scheduling the background
        task.
        """
        compiled = self._get_compiled(spec)
        # Fail-fast on bad seeds so the caller observes the ValueError
        # synchronously, not via an awaited handle.
        seed_state = self._build_initial_state(spec, seed_outputs)
        resolved_run_dir = _resolve_run_dir(run_context, run_dir)
        run_id = _get_run_id(run_context)
        execution_id = execution_id or make_execution_id(run_id, resolved_run_dir)

        handle = _GraphWorkflowExecution(
            execution_id=execution_id,
            workflow_id=spec.workflow_id,
            run_id=run_id,
        )

        async def _bg() -> None:
            try:
                workflow_deps = self._build_deps(
                    compiled,
                    run_context=run_context,
                    run_dir=resolved_run_dir,
                    config=config,
                    deps=deps,
                )
                result_state: WorkflowState = await _GRAPH.run(state=seed_state, deps=workflow_deps)
                handle._result = WorkflowResult(
                    status="failed" if result_state.failed else "completed",
                    outputs=result_state.results,
                    run_id=run_id,
                    execution_id=execution_id,
                )
            except Exception:
                handle._result = WorkflowResult(
                    status="failed",
                    outputs={},
                    run_id=handle.run_id,
                    execution_id=execution_id,
                )
                logger.exception(f"Background workflow {spec.name!r} failed")
            finally:
                handle._done_event.set()

        handle._task = asyncio.create_task(_bg())
        return handle

    # ── iter ─────────────────────────────────────────────────────────────────

    def iter(
        self,
        spec: Workflow,
        *,
        run_context: RunContextLike | None = None,
        run_dir: str | Path | None = None,
        config: JSONMapping | None = None,
        deps: UserDeps = None,
        seed_outputs: Mapping[str, TaskOutput] | None = None,
    ) -> Any:  # noqa: ANN401
        """Return an async context manager for step-by-step iteration.

        See :meth:`execute` for ``seed_outputs`` semantics.
        """
        compiled = self._get_compiled(spec)
        state = self._build_initial_state(spec, seed_outputs)
        resolved_run_dir = _resolve_run_dir(run_context, run_dir)
        workflow_deps = self._build_deps(
            compiled,
            run_context=run_context,
            run_dir=resolved_run_dir,
            config=config,
            deps=deps,
        )
        return _GRAPH.iter(state=state, deps=workflow_deps)

    # ── stream ───────────────────────────────────────────────────────────────

    def stream(
        self,
        spec: Workflow,
        *,
        run_context: RunContextLike | None = None,
        run_dir: str | Path | None = None,
        config: JSONMapping | None = None,
        deps: UserDeps = None,
        seed_outputs: Mapping[str, TaskOutput] | None = None,
    ) -> Any:  # noqa: ANN401
        """Alias for iter() — streaming Actor support in a future phase."""
        return self.iter(
            spec,
            run_context=run_context,
            run_dir=run_dir,
            config=config,
            deps=deps,
            seed_outputs=seed_outputs,
        )


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
