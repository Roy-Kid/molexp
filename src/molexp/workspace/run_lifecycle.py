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
from datetime import datetime
from typing import TYPE_CHECKING

from .models import ErrorInfo, ExecutionMetadata, ExecutionRecord, RunStatus

if TYPE_CHECKING:
    from .runcontext import RunContext


class RunLifecycle:
    """Enter/exit orchestration for a :class:`RunContext`."""

    def __init__(self, ctx: RunContext) -> None:
        self._ctx = ctx

    def enter(self) -> None:
        ctx = self._ctx
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        ctx._ctx_store.load_existing_results()
        self._apply_profile_metadata()
        self._claim_ownership()
        ctx.run._set_status(RunStatus.RUNNING)
        ctx._start_time = datetime.now()
        ctx._entered = True

        # Determine which execution attempt this is and record it.
        ctx._execution_id = ctx._explicit_execution_id or ctx._executions.next_execution_id()
        new_record = ExecutionRecord(
            execution_id=ctx._execution_id,
            started_at=ctx._start_time,
        )
        ctx.run._update_metadata(
            execution_history=[*ctx.run.metadata.execution_history, new_record]
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

    def exit(self, exc_type, exc_val, exc_tb) -> bool:  # noqa: ANN001
        ctx = self._ctx
        # ``enter()`` always runs first and assigns a non-None execution id.
        execution_id = ctx._execution_id
        assert execution_id is not None
        now = datetime.now()
        labels = self._labels_without_ownership()
        error_info: ErrorInfo | None = None
        if exc_type is None:
            workflow_status = ctx._ctx_store.context.status.get("run")
            final = RunStatus.FAILED if workflow_status == RunStatus.FAILED else RunStatus.SUCCEEDED
            ctx.run._update_metadata(
                status=final,
                finished_at=now,
                labels=labels,
                execution_history=ctx._executions.close_record(execution_id, final.value, now),
            )
        else:
            final = RunStatus.FAILED
            error_info = ErrorInfo(
                type=exc_type.__name__,
                message=str(exc_val),
                timestamp=now,
            )
            ctx.run._update_metadata(
                status=final,
                finished_at=now,
                labels=labels,
                error=error_info,
                execution_history=ctx._executions.close_record(execution_id, final.value, now),
            )
            ctx._assets.save_error_details(exc_type, exc_val, exc_tb)
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
        ctx._entered = False
        return False

    def _apply_profile_metadata(self) -> None:
        """Persist the active profile name / data / hash into RunMetadata."""
        ctx = self._ctx
        cfg = ctx._profile_config
        ctx.run._update_metadata(
            profile=cfg.name,
            config=cfg.to_dict(),
            config_hash=cfg.content_hash() if len(cfg) > 0 or cfg.name else None,
            labels=dict(ctx.run.metadata.labels),
        )

    def _claim_ownership(self) -> None:
        """Stamp the run with the current process identity.

        Stored in ``labels`` as ``pid`` / ``host`` / ``heartbeat``.  A later
        ``molexp run`` invocation can consult these to tell a live run from a
        zombie left behind by a crashed process.
        """
        ctx = self._ctx
        labels = dict(ctx.run.metadata.labels)
        labels["pid"] = str(os.getpid())
        labels["host"] = platform.node()
        labels["heartbeat"] = datetime.now().isoformat()
        ctx.run._update_metadata(labels=labels)

    def _labels_without_ownership(self) -> dict[str, str]:
        """Return labels with the ownership stamp removed."""
        labels = dict(self._ctx.run.metadata.labels)
        for key in ("pid", "host", "heartbeat"):
            labels.pop(key, None)
        return labels
