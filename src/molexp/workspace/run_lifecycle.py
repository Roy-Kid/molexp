"""``RunLifecycle`` — RunContext's enter/exit state machine.

Tier-1 collaborator of :class:`~molexp.workspace.run.RunContext` (see the
``workspace-slim-03-runcontext`` decomposition). Drives the
context-manager protocol: claim process ownership, stamp profile
metadata, flip run status, allocate the execution attempt, and on exit
close the record + persist results / error trace. It is the only
collaborator that *orchestrates* the others, so it holds a back-reference
to the facade and reaches the Tier-2/3 collaborators through it; the
reverse dependency (a store reaching back into the lifecycle) is
forbidden.

Exit-status resolution (run-recovery): success is a positive signal, never
the default on a previously failed run. With no exception, the final status
is (1) FAILED / SUCCEEDED when the workflow signalled via ``mark_failed`` /
``mark_succeeded``; (2) otherwise SUCCEEDED when the attempt recorded new
results or the run was not previously failed/cancelled (the documented
manual-driver contract); (3) otherwise a **no-op attempt** — the run keeps
its prior failed/cancelled status, the ExecutionRecord closes as
``"aborted"``, and ``metadata.error`` is preserved. A SUCCEEDED terminal
clears ``metadata.error``; an exception-free FAILED terminal also persists
``executions/<exec_id>/error.txt`` (the engine swallows task exceptions, so
this is the only chance to land the trace file).
"""

from __future__ import annotations

import os
import platform
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mollog import get_logger

from .models import ErrorInfo, ExecutionMetadata, ExecutionRecord, RunStatus

if TYPE_CHECKING:
    from .runcontext import RunContext

logger = get_logger(__name__)

#: Cadence of the ownership-heartbeat refresh while a run is executing.
#: Cross-host zombie reapers (see ``molexp.cli._common.reap_zombie_run``)
#: only reap a remote ``running`` run when this stamp is stale well beyond
#: the refresh cadence, so the two constants must stay far apart.
HEARTBEAT_INTERVAL_SECONDS = 30.0


def _split_error_text(raw: str) -> tuple[str, str]:
    """Split a ``"ExceptionType: message"`` string into ``(type, message)``.

    The workflow engine records task failures type-prefixed (e.g.
    ``"ZeroDivisionError: division by zero"``). Recover the type when the prefix
    is a Python-exception-style identifier; otherwise keep the whole string as
    the message under a generic ``"WorkflowError"`` type — never fabricate a
    specific type we did not actually see.
    """
    head, sep, tail = raw.partition(": ")
    if sep and head.isidentifier() and head[:1].isupper():
        return head, tail
    return "WorkflowError", raw


def _error_info_from_context(ctx: RunContext, now: datetime) -> ErrorInfo | None:
    """Build an :class:`ErrorInfo` from the message ``mark_failed`` stashed.

    ``mark_failed`` records ``context.errors["run"] = {"message": ...}`` when a
    task fails without the exception propagating out of ``execute()``. Returns
    ``None`` when no such message is present (nothing to persist).
    """
    run_err = ctx._ctx_store.context.errors.get("run")
    raw = run_err.get("message") if isinstance(run_err, dict) else run_err
    if not raw:
        return None
    etype, message = _split_error_text(str(raw))
    return ErrorInfo(type=etype, message=message, timestamp=now)


def _traceback_from_context(ctx: RunContext) -> str | None:
    """The formatted task traceback ``mark_failed`` stashed, if any.

    The workflow runtime forwards the swallowed task exception's formatted
    stack via ``mark_failed(..., traceback_text=…)``; it lands in
    ``context.errors["run"]["traceback"]``. ``None`` when the failure signal
    carried no traceback (e.g. a manual ``mark_failed("msg")``).
    """
    run_err = ctx._ctx_store.context.errors.get("run")
    if isinstance(run_err, dict):
        return run_err.get("traceback") or None
    return None


class RunLifecycle:
    """Enter/exit orchestration for a :class:`RunContext`."""

    def __init__(
        self,
        ctx: RunContext,
        *,
        heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self._ctx = ctx
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_stop: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._status_on_enter: RunStatus | None = None

    def enter(self) -> None:
        ctx = self._ctx
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        ctx._ctx_store.load_existing_results()
        ctx._ctx_store.reset_write_tracking()
        self._apply_profile_metadata()
        self._claim_ownership()
        # Remember the status this attempt started from: a signal-less no-op
        # attempt must restore it instead of defaulting to SUCCEEDED (bug 1).
        self._status_on_enter = ctx.run.read_ops().status
        ctx.run._set_status(RunStatus.RUNNING)
        ctx._start_time = datetime.now()
        ctx._entered = True

        # Determine which execution attempt this is and record it. When the
        # caller pre-allocated an execution_id that matches an existing record,
        # *reopen* that record in place (resume) — flip it back to running and
        # clear finished_at — instead of appending a new one. Any other case
        # (no id, or an id matching no record) appends a fresh record (rerun /
        # first attempt).
        explicit = ctx._explicit_execution_id
        history = ctx.run.read_ops().executions
        reopened = (
            next((r for r in history if r.execution_id == explicit), None)
            if explicit is not None
            else None
        )
        if reopened is not None:
            ctx._execution_id = reopened.execution_id
            running = reopened.model_copy(
                update={"status": RunStatus.RUNNING.value, "finished_at": None}
            )
            new_executions = tuple(
                running if r.execution_id == reopened.execution_id else r for r in history
            )
        else:
            ctx._execution_id = explicit or ctx._executions.next_execution_id()
            new_record = ExecutionRecord(
                execution_id=ctx._execution_id,
                started_at=ctx._start_time,
            )
            new_executions = (*history, new_record)
        # Record the active execution id + history in the OKF ``_ops`` hot-state
        # sidecar (wsokf-10). ``run.json`` (identity) carries no hot-state field.
        active_execution_id = ctx._execution_id
        ctx.run.update_ops(
            lambda state: state.model_copy(
                update={
                    "current_execution_id": active_execution_id,
                    "executions": new_executions,
                }
            )
        )
        ctx._executions.write_metadata(
            ExecutionMetadata(
                execution_id=ctx._execution_id,
                run_id=ctx.run.id,
                started_at=ctx._start_time,
                status=RunStatus.RUNNING.value,
            )
        )
        ctx._assets.append_run_log(f"execution started  exec_id={ctx._execution_id}")
        ctx._ctx_store.save()
        self._start_heartbeat()
        # Default-on, non-fatal workspace-timeline milestone (integration P0.3).
        self._emit_run_event("run.started", payload={"execution_id": ctx._execution_id})

    def exit(self, exc_type, exc_val, exc_tb) -> bool:  # noqa: ANN001
        # Stop the heartbeat first so it cannot race the terminal-status
        # writes below (the reaper must never see a fresh heartbeat on a
        # run whose status is already terminal-in-progress).
        self._stop_heartbeat()
        ctx = self._ctx
        # ``enter()`` always runs first and assigns a non-None execution id.
        execution_id = ctx._execution_id
        assert execution_id is not None
        now = datetime.now()
        error_info: ErrorInfo | None = None
        noop = False
        if exc_type is None:
            # Three-state resolution (bug 1): FAILED / SUCCEEDED are explicit
            # workflow signals (``mark_failed`` / ``mark_succeeded``); with NO
            # signal, a previously failed/cancelled run that also recorded no
            # new results is a no-op attempt — it keeps its prior status and
            # its ExecutionRecord closes as "aborted". Success is never the
            # default on a run that already failed. A run that was never
            # failed keeps the legacy contract: a clean, exception-free
            # attempt (manual-driver / artifact-only usage) resolves to
            # SUCCEEDED.
            workflow_status = ctx._ctx_store.context.status.get("run")
            status_on_enter = self._status_on_enter
            if workflow_status == RunStatus.FAILED:
                final = RunStatus.FAILED
                # The engine caught the task exception into a FAILED workflow
                # status (it does NOT re-raise, so the run stays resumable) and
                # stashed the message on the context via ``mark_failed``. Lift it
                # into the workspace-owned canonical record, so run.json /
                # execution.json don't silently carry ``error: null`` while the
                # reason lives only in the workflow-layer document.
                error_info = _error_info_from_context(ctx, now)
                if error_info is not None:
                    ctx.run._update_metadata(error=error_info)
                    # The exception never propagated (the engine swallowed it),
                    # so this is the only chance to land error.txt (bug 3) —
                    # including the real traceback the runtime forwarded
                    # through ``mark_failed(..., traceback_text=…)``.
                    ctx._assets.save_error_report(
                        error_type=error_info.type,
                        message=error_info.message,
                        traceback_text=_traceback_from_context(ctx),
                    )
            elif workflow_status == RunStatus.SUCCEEDED:
                final = RunStatus.SUCCEEDED
            elif (
                status_on_enter in (RunStatus.FAILED, RunStatus.CANCELLED)
                and not ctx._ctx_store.wrote_results
            ):
                final = status_on_enter
                noop = True
            else:
                final = RunStatus.SUCCEEDED
        else:
            final = RunStatus.FAILED
            error_info = ErrorInfo(
                type=exc_type.__name__,
                message=str(exc_val),
                timestamp=now,
            )
            # ``error`` is identity/diagnostic — it stays in run.json (wsokf-10).
            ctx.run._update_metadata(error=error_info)
            ctx._assets.save_error_details(exc_type, exc_val, exc_tb)
        if final is RunStatus.SUCCEEDED and ctx.run.metadata.error is not None:
            # A successful terminal state must not keep describing an old
            # failure (bug 2) — the canonical record tells the truth.
            ctx.run._update_metadata(error=None)
        # A no-op attempt closes its record as "aborted" — it neither
        # succeeded nor failed; the run-level status stays what it was.
        record_status = "aborted" if noop else final.value
        # Terminal hot-state — status / finished_at / closed executions / cleared
        # ownership — is written solely to the OKF ``_ops`` sidecar (wsokf-10).
        closed_executions = tuple(ctx._executions.close_record(execution_id, record_status, now))
        ctx.run.update_ops(
            lambda state: state.model_copy(
                update={
                    "status": final,
                    "finished_at": now,
                    "executions": closed_executions,
                    "owner_pid": None,
                    "owner_host": None,
                    "heartbeat_at": None,
                }
            )
        )
        ctx._executions.update_metadata(
            execution_id,
            finished_at=now,
            status=record_status,
            error=error_info,
        )
        ctx._assets.append_run_log(
            f"execution finished exec_id={ctx._execution_id}  status={record_status}"
        )
        ctx._ctx_store.save()
        # Low-frequency git checkpoint at the Execution-settled boundary: one
        # commit per settled execution, and only when the projection DB already
        # exists (opt-in by existence). Best-effort — never breaks the run.
        self._checkpoint_git_on_settle()
        # Default-on, non-fatal workspace-timeline milestone (integration P0.3).
        # A no-op attempt changed nothing — it emits no completion/failure event.
        if not noop:
            settled = "run.completed" if final is RunStatus.SUCCEEDED else "run.failed"
            self._emit_run_event(
                settled, payload={"status": final.value, "execution_id": execution_id}
            )
        ctx._entered = False
        return False

    def _checkpoint_git_on_settle(self) -> None:
        """Trigger the opt-in, best-effort git checkpoint for this settled run."""
        from molexp.workspace.git_projection import checkpoint_run_on_settle

        checkpoint_run_on_settle(self._ctx.run)

    def _emit_run_event(self, event_type, *, payload) -> None:  # noqa: ANN001
        """Append a default-on, non-fatal run milestone to the workspace event spine.

        Resolves the workspace root from the run (the same
        ``run.experiment.project.workspace`` path :meth:`_checkpoint_git_on_settle`
        uses) and delegates to :func:`molexp.workspace.events.emit_workspace_event`,
        which swallows any failure (the timeline never breaks a run).
        """
        from molexp.workspace.events import emit_workspace_event

        run = self._ctx.run
        emit_workspace_event(
            run.experiment.project.workspace.resolve(),
            event_type,
            "run-lifecycle",
            payload=payload,
            refs=[run.id],
        )

    def _apply_profile_metadata(self) -> None:
        """Persist the active profile name / data / hash into RunMetadata."""
        ctx = self._ctx
        cfg = ctx._profile_config
        ctx.run._update_metadata(
            profile=cfg.name,
            config=cfg.to_dict(),
            config_hash=cfg.content_hash() if len(cfg) > 0 or cfg.name else None,
        )

    def _claim_ownership(self) -> None:
        """Stamp the run with the current process identity in the ``_ops`` sidecar.

        Stored as ``owner_pid`` / ``owner_host`` / ``heartbeat_at`` (aware-UTC)
        on :class:`RunOpsState` (wsokf-10).  A later ``molexp run`` invocation
        can consult these to tell a live run from a zombie left behind by a
        crashed process.
        """
        ctx = self._ctx
        now = datetime.now(UTC)
        pid = os.getpid()
        host = platform.node()
        ctx.run.update_ops(
            lambda state: state.model_copy(
                update={
                    "owner_pid": pid,
                    "owner_host": host,
                    "heartbeat_at": now,
                }
            )
        )

    # ── Heartbeat ────────────────────────────────────────────────────────
    #
    # The ownership stamp written by ``_claim_ownership`` includes a
    # ``heartbeat_at`` timestamp. Same-host reapers can check the pid directly,
    # but cross-host observers (molq / SLURM submissions are the core
    # scenario) have only this timestamp to tell a live remote run from a
    # zombie — so it must be refreshed while the run executes.

    def _start_heartbeat(self) -> None:
        """Spawn the daemon thread that re-stamps ``heartbeat_at``."""
        stop = threading.Event()
        thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(stop,),
            name=f"molexp-heartbeat-{self._ctx.run.id}",
            daemon=True,
        )
        self._heartbeat_stop = stop
        self._heartbeat_thread = thread
        thread.start()

    def _stop_heartbeat(self) -> None:
        """Signal the heartbeat thread to exit and wait briefly for it."""
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5.0)
        self._heartbeat_stop = None
        self._heartbeat_thread = None

    def _heartbeat_loop(self, stop: threading.Event) -> None:
        while not stop.wait(self._heartbeat_interval):
            try:
                self.refresh_heartbeat()
            except Exception:
                # Never let a display/metadata hiccup kill the worker;
                # a missed beat only delays staleness detection.
                logger.debug(f"heartbeat refresh failed for run {self._ctx.run.id}", exc_info=True)

    def refresh_heartbeat(self) -> None:
        """Re-stamp the run's heartbeat in the OKF ``_ops`` sidecar (aware-UTC).

        The live-run heartbeat lives on :attr:`RunOpsState.heartbeat_at` (an
        aware-UTC timestamp) so cross-host staleness comparisons are tz-correct
        (wsokf-07/wsokf-10) — a single ``update_ops`` read-modify-write of
        ``_ops/run.json``. ``run.json`` (identity) is never touched.

        A run whose ``_ops/run.json`` has not been written yet (first beat
        before the lifecycle claimed ownership) is left untouched.
        """
        run = self._ctx.run
        ops_path = run._fs.join(run.run_dir, "_ops", "run.json")
        if not run._fs.exists(ops_path):
            return
        now = datetime.now(UTC)
        run.update_ops(lambda state: state.model_copy(update={"heartbeat_at": now}))
