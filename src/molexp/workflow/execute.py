"""One-step tracked execution — run a workflow against a workspace ``Run``.

``execute_run(workflow, run)`` folds the driver dance (``run.start()`` context
+ ``WorkflowRuntime().execute(..., run_context=ctx)`` + asyncio plumbing) into
a single call on the **same execution path as** ``molexp run``: the
``RunContext`` lifecycle owns the status machine, the ``ops/run.json``
hot-state sidecar and the ownership heartbeat; the workflow engine owns
scheduling, caching and node-level persistence. Nothing here is a second
path — it is the CLI's in-process handler, made importable.

Verb selection follows the canonical run-status law (see CLAUDE.md); the two
keywords mirror the CLI exactly — ``rerun`` is ``--rerun`` (fresh attempt, no
seeding) and ``fresh`` is ``--fresh`` (bypass the content-addressed cache
read; only meaningful with ``rerun=True``):

======================  ====================  =================================
run status              ``rerun=False``       ``rerun=True``
======================  ====================  =================================
``pending``             run (first attempt)   run (first attempt)
``failed``/``cancelled``  resume (reopen +      rerun (new ``exec-<id>-N``,
                        seed completed nodes)  no seeding)
``succeeded``           :class:`RunNotExecutableError` — done is done
``running``             :class:`RunNotExecutableError` — cancel it first
======================  ====================  =================================

A task failure raises :class:`RunFailedError` (carrying the partial
``WorkflowResult``) *after* the failed state has been persisted — loud like
the CLI's non-zero exit, never a silently-failed return value.

This module also registers the workspace ``set_run_executor`` inversion seam
at import time, which is what makes ``Run.execute`` / ``Run.aexecute`` and
``RunSet.execute`` work without the workspace layer ever importing workflow.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from molexp.workspace.run import RETRYABLE_STATUSES, RunStatus, set_run_executor

from ._engine.persistence import seed_from_execution
from ._engine.runtime import WorkflowRuntime
from .compiled import CompiledWorkflow
from .compiler import WorkflowCompiler

if TYPE_CHECKING:
    from molexp.profile import ProfileConfig
    from molexp.workspace.run import Run

    from .types import WorkflowResult

__all__ = [
    "RunFailedError",
    "RunNotExecutableError",
    "aexecute_run",
    "execute_run",
]


class RunNotExecutableError(RuntimeError):
    """The run's current status is outside ``execute_run``'s verb domain."""


class RunFailedError(RuntimeError):
    """A task failed during ``execute_run``; the run is persisted as failed.

    Carries the partial :class:`~molexp.workflow.WorkflowResult` on
    ``.result`` (completed upstream outputs remain readable), so callers that
    aggregate failures — e.g. ``RunSet.execute`` — keep the honest summary.
    """

    def __init__(self, message: str, *, result: WorkflowResult) -> None:
        super().__init__(message)
        self.result = result


def _ensure_compiled(workflow: object) -> CompiledWorkflow:
    """Normalize *workflow* to a :class:`CompiledWorkflow` (auto-compile)."""
    if isinstance(workflow, CompiledWorkflow):
        return workflow
    if isinstance(workflow, WorkflowCompiler):
        return workflow.compile()
    raise TypeError(
        f"execute_run expects a CompiledWorkflow or a WorkflowCompiler, "
        f"got {type(workflow).__name__}"
    )


def _select_verb(run: Run, *, rerun: bool) -> tuple[str | None, dict | None]:
    """Map the run's status (+ ``rerun``) to ``(execution_id, seed_outputs)``.

    ``(None, None)`` means a fresh attempt (first run / explicit rerun); a
    non-``None`` execution id reopens that attempt with its completed nodes
    seeded. Raises :class:`RunNotExecutableError` outside the verb domain.
    """
    status = run.status
    if status == RunStatus.RUNNING.value:
        raise RunNotExecutableError(
            f"run {run.id} is running — one Run carries one ownership stamp and "
            f"one status, so a second concurrent execution is never started. "
            f"cancel it first (run.cancel() / `molexp runs cancel`)."
        )
    if status == RunStatus.SUCCEEDED.value:
        raise RunNotExecutableError(
            f"run {run.id} already succeeded — read results via run.get_result(...) "
            f"or result.outputs. Re-executing a succeeded run is not a molexp "
            f"operation; declare a run with different params instead."
        )
    if not rerun and status in RETRYABLE_STATUSES:
        # resume: reopen the last execution, seed its completed nodes. The
        # no-fallback semantics live in ``seed_from_execution``.
        return seed_from_execution(run)
    return None, None


def _raise_if_failed(run: Run, result: WorkflowResult) -> WorkflowResult:
    if result.status == "succeeded":
        return result
    error = getattr(run.metadata, "error", None)
    detail = f"{error.type}: {error.message}" if error is not None else "see the run's error logs"
    raise RunFailedError(
        f"run {run.id} failed — {detail} "
        f"(run dir: {run.run_dir}; retry with execute_run(...) to resume, "
        f"or rerun=True to re-execute from the top)",
        result=result,
    )


async def aexecute_run(
    workflow: object,
    run: Run,
    *,
    rerun: bool = False,
    fresh: bool = False,
    profile_config: ProfileConfig | None = None,
) -> WorkflowResult:
    """Async one-step tracked execution. See :func:`execute_run`."""
    if fresh and not rerun:
        raise ValueError(
            "fresh=True bypasses the cache for an explicit re-execution and "
            "requires rerun=True (mirroring `molexp run --rerun --fresh`)."
        )
    compiled = _ensure_compiled(workflow)
    execution_id, seed_outputs = _select_verb(run, rerun=rerun)
    with run.start(profile_config, execution_id=execution_id) as ctx:
        result = await WorkflowRuntime().execute(
            compiled,
            run_context=ctx,
            execution_id=execution_id,
            seed_outputs=seed_outputs,
            bypass_cache=fresh,
        )
    return _raise_if_failed(run, result)


def execute_run(
    workflow: object,
    run: Run,
    *,
    rerun: bool = False,
    fresh: bool = False,
    profile_config: ProfileConfig | None = None,
) -> WorkflowResult:
    """Execute *workflow* against *run* in one step and return the result.

    The synchronous facade over :func:`aexecute_run` — owns the event loop via
    ``asyncio.run``. Args:

        workflow: A ``CompiledWorkflow``, or an uncompiled
            ``WorkflowCompiler`` (compiled automatically).
        run: The workspace :class:`~molexp.workspace.run.Run` to execute
            against (status machine / ``ops`` sidecar / heartbeat are driven
            by its ``RunContext`` lifecycle, exactly as under ``molexp run``).
        rerun: ``True`` forces a fresh attempt (new ``exec-<run_id>-N``, no
            seeding) instead of the default resume-on-retryable behaviour —
            the CLI's ``--rerun``.
        fresh: ``True`` additionally bypasses the content-addressed cache
            *read* for that attempt (results are still written back). Requires
            ``rerun=True`` — the CLI's ``--rerun --fresh``.
        profile_config: Optional molcfg profile applied to the run.

    Returns:
        The ``WorkflowResult`` — ``result.outputs`` maps task name → output.

    Raises:
        RunFailedError: A task failed (state persisted first).
        RunNotExecutableError: Status outside the verb domain (see module doc).
        RuntimeError: Called from inside a running event loop — await
            :func:`aexecute_run` there instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "execute_run() was called from inside a running event loop; "
            "await aexecute_run(...) (or run.aexecute(...)) instead."
        )
    return asyncio.run(
        aexecute_run(workflow, run, rerun=rerun, fresh=fresh, profile_config=profile_config)
    )


# ── Workspace inversion seam ─────────────────────────────────────────────────


class _WorkspaceRunExecutor:
    """Implements ``molexp.workspace.run.RunWorkflowExecutor``.

    ``workflow=None`` resolves the experiment's bound workflow from
    :data:`~molexp.workflow.binding.default_binding_registry` (the store
    ``Experiment.run`` / ``Experiment.sweep`` populate) — and fails fast when
    nothing is bound, never falling back.
    """

    @staticmethod
    def _resolve(run: Run, workflow: object | None) -> object:
        if workflow is not None:
            return workflow
        from .binding import default_binding_registry

        bound = default_binding_registry.for_experiment(run.experiment)
        if bound is None:
            raise RuntimeError(
                f"run {run.id}: no workflow bound to experiment "
                f"{run.experiment.id!r} — pass workflow=... or declare it via "
                f"experiment.sweep(workflow, params=...) / experiment.run(workflow, ...)."
            )
        return bound

    def execute(
        self, run: Run, workflow: object | None, *, rerun: bool = False, fresh: bool = False
    ) -> object:
        return execute_run(self._resolve(run, workflow), run, rerun=rerun, fresh=fresh)

    async def aexecute(
        self, run: Run, workflow: object | None, *, rerun: bool = False, fresh: bool = False
    ) -> object:
        return await aexecute_run(self._resolve(run, workflow), run, rerun=rerun, fresh=fresh)


# Wire the seam at import time so ``run.execute(workflow)`` works as soon as
# the workflow layer is loaded (the caller necessarily imported it to build
# ``workflow``), without workspace ever importing this layer.
set_run_executor(_WorkspaceRunExecutor())
