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

    def enter(self) -> None:
        ctx = self._ctx
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        ctx._ctx_store.load_existing_results()
        self._apply_profile_metadata()
        self._claim_ownership()
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
        if exc_type is None:
            workflow_status = ctx._ctx_store.context.status.get("run")
            final = RunStatus.FAILED if workflow_status == RunStatus.FAILED else RunStatus.SUCCEEDED
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
        # Terminal hot-state — status / finished_at / closed executions / cleared
        # ownership — is written solely to the OKF ``_ops`` sidecar (wsokf-10).
        closed_executions = tuple(ctx._executions.close_record(execution_id, final.value, now))
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
            status=final.value,
            error=error_info,
        )
        ctx._assets.append_run_log(
            f"execution finished exec_id={ctx._execution_id}  status={final.value}"
        )
        ctx._ctx_store.save()
        # Low-frequency git checkpoint at the Execution-settled boundary: one
        # commit per settled execution, and only when the projection DB already
        # exists (opt-in by existence). Best-effort — never breaks the run.
        self._checkpoint_git_on_settle()
        ctx._entered = False
        return False

    def _checkpoint_git_on_settle(self) -> None:
        """Trigger the opt-in, best-effort git checkpoint for this settled run."""
        from molexp.workspace.git_projection import checkpoint_run_on_settle

        checkpoint_run_on_settle(self._ctx.run)

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
