"""GraphWorkflowRuntime: concrete WorkflowRuntime backed by pydantic-graph.

Implements the WorkflowRuntime interface for all three modes:
- execute(): run to completion, return WorkflowResult
- start(): launch in background, return WorkflowExecution handle
- resume(): continue from persisted snapshot
- stream()/iter(): async step-by-step iteration
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from mollog import get_logger

from molexp.config import ProfileConfig

from ..runtime import WorkflowRuntime
from ..types import WorkflowExecution, WorkflowResult
from .compiler import CompiledWorkflow, WorkflowGraphCompiler
from .node import WorkflowStep
from .persistence import RunStorePersistence
from .state import WorkflowDeps, WorkflowState

logger = get_logger(__name__)

_compiler = WorkflowGraphCompiler()


def _get_run_dir(run: Any) -> Path | None:
    """Extract run directory from a molexp Run object if available."""
    if run is None:
        return None
    for attr in ("run_dir", "path", "root"):
        val = getattr(run, attr, None)
        if val is not None:
            return Path(val)
    return None


def _get_run_id(run: Any) -> str | None:
    """Extract a stable run identifier from a molexp Run-like object."""
    if run is None:
        return None
    return getattr(run, "id", getattr(run, "run_id", None))


def _make_execution_id(run_id: str | None, run_dir: Path | None) -> str:
    """Build an execution ID derived from run_id.

    First execution: ``exec-{run_id}``
    Retries:         ``exec-{run_id}-2``, ``exec-{run_id}-3``, …

    Falls back to a random ID when *run_id* is not available.
    """
    if run_id is None:
        return f"exec-{uuid.uuid4().hex[:8]}"

    base = f"exec-{run_id}"
    if run_dir is None:
        return base

    exec_root = run_dir / "execution"
    if not exec_root.exists():
        return base

    existing = [p for p in exec_root.iterdir() if p.name.startswith(base)]
    if not existing:
        return base
    return f"{base}-{len(existing) + 1}"


class GraphWorkflowRuntime(WorkflowRuntime):
    """Workflow runtime powered by pydantic-graph.

    Caches compiled graphs by workflow_id so repeated executions of the
    same spec skip recompilation.
    """

    def __init__(self) -> None:
        self._compiled: dict[str, CompiledWorkflow] = {}

    def _get_compiled(self, spec: Any) -> CompiledWorkflow:
        wf_id = spec.workflow_id
        if wf_id not in self._compiled:
            self._compiled[wf_id] = _compiler.compile(spec)
        return self._compiled[wf_id]

    @staticmethod
    def _build_deps(
        compiled: CompiledWorkflow,
        *,
        run: Any,
        run_context: Any,
        profile_config: ProfileConfig | None,
        kwargs: dict[str, Any],
    ) -> tuple[WorkflowDeps, Path | None]:
        """Build WorkflowDeps with optional remote executor from kwargs.

        When *run_context* is provided, its attached :attr:`config`
        takes precedence over *profile_config* — the context was
        constructed with a fixed ``ProfileConfig`` and must be honoured.
        """
        user_deps = kwargs.get("deps")
        submitors = kwargs.get("submitors")
        run_dir = (
            Path(run_context.work_dir)
            if run_context is not None
            else _get_run_dir(run)
        )

        effective_config = (
            run_context.config if run_context is not None else profile_config
        )

        remote_executor = None
        if submitors:
            from ..remote import RemoteStepExecutor
            remote_executor = RemoteStepExecutor(submitors)

        deps = compiled.make_deps(
            run=run,
            run_context=run_context,
            config=effective_config,
            user_deps=user_deps,
            remote_executor=remote_executor,
            run_dir=run_dir,
        )
        return deps, run_dir

    @staticmethod
    def _set_run_status(run_context: Any, *, failed: bool, persist: bool) -> None:
        if run_context is None or not hasattr(run_context, "context"):
            return

        from molexp.workspace.run import RunStatus

        status = RunStatus.FAILED if failed else RunStatus.SUCCEEDED

        run_context.context.status["run"] = status
        if persist and hasattr(run_context, "_save_context"):
            run_context._save_context()

    @asynccontextmanager
    async def _execution_scope(
        self,
        *,
        run: Any,
        run_context: Any,
        profile_config: ProfileConfig | None,
    ) -> AsyncGenerator[Any, None]:
        if run is not None and run_context is not None:
            raise ValueError("Pass either run or run_context, not both")

        if run_context is not None:
            if profile_config is not None:
                raise ValueError(
                    "Cannot combine run_context with profile_config.  "
                    "The profile must be fixed on the RunContext at construction "
                    "time — late-binding is not permitted."
                )
            yield run_context
            return

        if run is not None:
            from molexp.workspace.run import RunContext as WorkspaceRunContext
            managed_ctx = WorkspaceRunContext(run, profile_config=profile_config)
            with managed_ctx:
                yield managed_ctx
            return

        yield None

    # ── execute ──────────────────────────────────────────────────────────────

    async def execute(
        self,
        spec: Any,
        run: Any = None,
        run_context: Any = None,
        *,
        profile_config: ProfileConfig | None = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Run the workflow to completion and return a WorkflowResult."""
        if run is not None and run_context is not None:
            raise ValueError("Pass either run or run_context, not both")
        if run_context is not None and profile_config is not None:
            raise ValueError(
                "Cannot combine run_context with profile_config.  "
                "The profile must be fixed on the RunContext at construction time."
            )

        compiled = self._get_compiled(spec)
        # Derive execution_id from run_id so all layers share the same base ID.
        _early_run = run_context.run if run_context is not None else run
        _early_run_id = _get_run_id(_early_run)
        _early_run_dir = (
            Path(run_context.work_dir) if run_context is not None
            else _get_run_dir(_early_run)
        )
        execution_id = _make_execution_id(_early_run_id, _early_run_dir)
        owner_supplied_context = run_context

        try:
            async with self._execution_scope(
                run=run,
                run_context=run_context,
                profile_config=profile_config,
            ) as active_run_context:
                effective_run = (
                    active_run_context.run
                    if active_run_context is not None
                    else run
                )
                deps, run_dir = self._build_deps(
                    compiled,
                    run=effective_run,
                    run_context=active_run_context,
                    profile_config=profile_config,
                    kwargs=kwargs,
                )
                state = WorkflowState()
                run_id = _get_run_id(deps.run)

                persistence = (
                    RunStorePersistence(run_dir=run_dir, execution_id=execution_id)
                    if run_dir is not None
                    else None
                )

                if persistence is not None:
                    run_result = await compiled.graph.run(
                        WorkflowStep(0),
                        state=state,
                        deps=deps,
                        persistence=persistence,
                    )
                else:
                    run_result = await compiled.graph.run(
                        WorkflowStep(0),
                        state=state,
                        deps=deps,
                    )
                result_state = run_result.output

                if result_state.failed:
                    self._set_run_status(
                        active_run_context,
                        failed=True,
                        persist=owner_supplied_context is not None,
                    )
                    return WorkflowResult(
                        status="failed",
                        outputs=result_state.step_outputs,
                        run_id=run_id,
                        execution_id=execution_id,
                    )

                self._set_run_status(
                    active_run_context,
                    failed=False,
                    persist=owner_supplied_context is not None,
                )
                return WorkflowResult(
                    status="completed",
                    outputs=result_state.step_outputs,
                    run_id=run_id,
                    execution_id=execution_id,
                )
        except Exception:
            self._set_run_status(
                owner_supplied_context,
                failed=True,
                persist=owner_supplied_context is not None,
            )
            logger.exception(f"Workflow {spec.name!r} execution failed")
            return WorkflowResult(
                status="failed",
                outputs={},
                run_id=_get_run_id(
                    owner_supplied_context.run
                    if owner_supplied_context is not None
                    else run
                ),
                execution_id=execution_id,
            )

    # ── start ────────────────────────────────────────────────────────────────

    async def start(
        self,
        spec: Any,
        run: Any = None,
        run_context: Any = None,
        *,
        profile_config: ProfileConfig | None = None,
        **kwargs: Any,
    ) -> WorkflowExecution:
        """Launch workflow as background asyncio task."""
        if run is not None and run_context is not None:
            raise ValueError("Pass either run or run_context, not both")
        if run_context is not None and profile_config is not None:
            raise ValueError(
                "Cannot combine run_context with profile_config.  "
                "The profile must be fixed on the RunContext at construction time."
            )

        compiled = self._get_compiled(spec)
        owner_supplied_context = run_context
        effective_run = (
            run_context.run
            if run_context is not None
            else run
        )
        run_id = _get_run_id(effective_run)
        _early_run_dir = (
            Path(run_context.work_dir) if run_context is not None
            else _get_run_dir(effective_run)
        )
        execution_id = _make_execution_id(run_id, _early_run_dir)

        handle = _GraphWorkflowExecution(
            execution_id=execution_id,
            workflow_id=spec.workflow_id,
            run_id=run_id,
        )

        async def _bg() -> None:
            try:
                async with self._execution_scope(
                    run=run,
                    run_context=run_context,
                    profile_config=profile_config,
                ) as active_run_context:
                    effective_run = (
                        active_run_context.run
                        if active_run_context is not None
                        else run
                    )
                    deps, _ = self._build_deps(
                        compiled,
                        run=effective_run,
                        run_context=active_run_context,
                        profile_config=profile_config,
                        kwargs=kwargs,
                    )
                    state = WorkflowState()
                    run_result = await compiled.graph.run(
                        WorkflowStep(0), state=state, deps=deps
                    )
                    result_state = run_result.output
                    self._set_run_status(
                        active_run_context,
                        failed=result_state.failed,
                        persist=owner_supplied_context is not None,
                    )
                    handle._result = WorkflowResult(
                        status="failed" if result_state.failed else "completed",
                        outputs=result_state.step_outputs,
                        run_id=_get_run_id(effective_run),
                        execution_id=execution_id,
                    )
            except Exception:
                self._set_run_status(
                    owner_supplied_context,
                    failed=True,
                    persist=owner_supplied_context is not None,
                )
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

    # ── resume ───────────────────────────────────────────────────────────────

    async def resume(
        self, spec: Any, run: Any, execution_id: str, **kwargs: Any
    ) -> WorkflowExecution:
        """Resume a workflow from persisted state."""
        run_dir = _get_run_dir(run)
        if run_dir is None:
            raise ValueError("Cannot resume workflow without a run directory")

        compiled = self._get_compiled(spec)
        # On resume we rebuild the ProfileConfig from the persisted
        # run.metadata so user tasks see the same ctx.config data.
        profile_cfg: ProfileConfig | None = None
        meta = getattr(run, "metadata", None)
        if meta is not None:
            profile_cfg = ProfileConfig(
                getattr(meta, "config", {}) or {},
                name=getattr(meta, "profile", None),
            )
        deps, _ = self._build_deps(
            compiled,
            run=run,
            run_context=None,
            profile_config=profile_cfg,
            kwargs=kwargs,
        )
        persistence = RunStorePersistence(run_dir=run_dir, execution_id=execution_id)

        handle = _GraphWorkflowExecution(
            execution_id=execution_id,
            workflow_id=spec.workflow_id,
            run_id=_get_run_id(run),
        )

        async def _bg() -> None:
            try:
                async with compiled.graph.iter_from_persistence(
                    persistence=persistence,
                    deps=deps,
                ) as run_ctx:
                    async for _node in run_ctx:
                        pass
                    result_state = run_ctx.result.output if run_ctx.result else WorkflowState(failed=True, error="No result")

                handle._result = WorkflowResult(
                    status="failed" if result_state.failed else "completed",
                    outputs=result_state.step_outputs,
                    run_id=handle.run_id,
                    execution_id=execution_id,
                )
            except Exception:
                handle._result = WorkflowResult(
                    status="failed",
                    outputs={},
                    run_id=handle.run_id,
                    execution_id=execution_id,
                )
                logger.exception(f"Resume of workflow {spec.name!r} failed")
            finally:
                handle._done_event.set()

        handle._task = asyncio.create_task(_bg())
        return handle

    # ── iter ─────────────────────────────────────────────────────────────────

    def iter(self, spec: Any, run: Any = None, **kwargs: Any) -> Any:
        """Return an async context manager for step-by-step iteration."""
        effective_run_context = kwargs.get("run_context")
        if run is not None and effective_run_context is not None:
            raise ValueError("Pass either run or run_context, not both")

        compiled = self._get_compiled(spec)
        effective_run = (
            effective_run_context.run
            if effective_run_context is not None
            else run
        )
        deps, _ = self._build_deps(
            compiled,
            run=effective_run,
            run_context=effective_run_context,
            profile_config=kwargs.get("profile_config"),
            kwargs=kwargs,
        )
        state = WorkflowState()
        return compiled.graph.iter(WorkflowStep(0), state=state, deps=deps)

    # ── stream ───────────────────────────────────────────────────────────────

    def stream(self, spec: Any, run: Any = None, **kwargs: Any) -> Any:
        """Alias for iter() — streaming Actor support in a future phase."""
        return self.iter(spec, run=run, **kwargs)


class _GraphWorkflowExecution(WorkflowExecution):
    """Concrete WorkflowExecution returned by start() / resume()."""

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
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._done_event.set()
