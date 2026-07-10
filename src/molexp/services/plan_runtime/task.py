"""``PlanTask`` — one background PlanMode pipeline run for the server.

A plan task is one-shot (run PlanMode once on a content-addressed Run), so —
unlike the agent-session runtime — it needs no session/turn split: the task IS
the background ``asyncio.Task`` plus its coarse status. On success it persists
the generated workflow onto the experiment so the UI graph renderer shows it.

Approvals are **never** auto-granted: ``PlanMode()`` runs with no approver, so
each gate resolves store-first (a grant recorded in the run's approval store
replays) and otherwise suspends — the task lands ``waiting_approval`` with the
pending requests kept on it, the approvals inbox lists them, and a granted
decision re-drives the pipeline via :meth:`PlanTask.resume` (the stage ledger
skips completed stages; the gate passes on the stored grant).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Literal

from mollog import get_logger

if TYPE_CHECKING:
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.schemas import ApprovalRequest, ModeResult
    from molexp.services.plan_runtime.materialize import PlanRecordOutcome
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.models import ComputeTarget
    from molexp.workspace.run import Run

__all__ = ["PlanTask", "PlanTaskStatus"]

_LOG = get_logger(__name__)

PlanTaskStatus = Literal["running", "completed", "failed", "cancelled", "waiting_approval"]


class PlanTask:
    """A single background PlanMode run, its status, and its result."""

    def __init__(
        self,
        *,
        task_id: str,
        run: Run,
        experiment: Experiment,
        draft: str,
        model: str,
        created_at: str,
        ground: bool = True,
        workspace_root: str = "",
        execute: bool = False,
        compute_target: ComputeTarget | None = None,
    ) -> None:
        self.task_id = task_id
        self.run = run
        self.experiment = experiment
        self.draft = draft
        self.model = model
        self.created_at = created_at
        self.ground = ground
        self.workspace_root = workspace_root
        self.execute = execute
        self.compute_target = compute_target
        self.status: PlanTaskStatus = "running"
        self.error: BaseException | None = None
        self.workflow_persisted = False
        self.record_outcome: PlanRecordOutcome | None = None
        self.result: ModeResult | None = None
        self.pending_requests: list[ApprovalRequest] = []
        self._gateway: AgentGateway | None = None
        self._task: asyncio.Task[None] | None = None

    @classmethod
    def start(
        cls,
        *,
        task_id: str,
        run: Run,
        experiment: Experiment,
        draft: str,
        model: str,
        created_at: str,
        gateway: AgentGateway,
        ground: bool = True,
        workspace_root: str = "",
        execute: bool = False,
        compute_target: ComputeTarget | None = None,
    ) -> PlanTask:
        """Build a task and spawn its background PlanMode run.

        ``execute=True`` appends the real-execution tail — the driver runs as
        an executor subprocess **of the serving host** (exactly what the CLI
        does on its host); ``compute_target`` only feeds the step-9
        *descriptive* execution report, it never schedules to molq. Every
        gate (including ``approve_execution``) suspends into the approvals
        inbox — server-side execution is unreachable without granted
        decisions.
        """
        task = cls(
            task_id=task_id,
            run=run,
            experiment=experiment,
            draft=draft,
            model=model,
            created_at=created_at,
            ground=ground,
            workspace_root=workspace_root,
            execute=execute,
            compute_target=compute_target,
        )
        task._gateway = gateway
        task._sync_status()  # visible in the Agents hub from launch, not completion
        task._task = asyncio.create_task(task._drive(gateway))
        return task

    async def _drive(self, gateway: AgentGateway) -> None:
        from molexp.harness import ApprovalPendingError, PlanMode

        from .drive import drive_plan_mode
        from .materialize import materialize_plan_records

        try:
            # Resolve molmcp grounding first (loud on miss, never silent) so the
            # binder picks real capabilities and ValidateBoundWorkflow checks them.
            capability_registry = None
            if self.ground:
                from molexp.mcp_capabilities import aresolve_capability_registry

                capability_registry = await aresolve_capability_registry(
                    self.workspace_root,
                    task=self.draft,
                )
            # drive_plan_mode wraps the pipeline in the run lifecycle so the
            # plan Run's status is honest (running -> succeeded | failed) —
            # the same shared path `molexp plan` uses.
            self.result = await drive_plan_mode(
                PlanMode(execute=self.execute, compute_target=self.compute_target),
                run=self.run,
                user_input=self.draft,
                gateway=gateway,
                capability_registry=capability_registry,
            )
            # Persist the workflow IR + record the Agents-tab session and
            # Knowledge records. Shared with `molexp plan` (CLI) so the Python
            # and UI paths land identical workspace state. Blocking I/O — offloaded.
            outcome = await asyncio.to_thread(
                lambda: materialize_plan_records(
                    run=self.run,
                    experiment=self.experiment,
                    workspace_root=self.workspace_root,
                    task_id=self.task_id,
                    draft=self.draft,
                    model=self.model,
                )
            )
            self.record_outcome = outcome
            self.workflow_persisted = outcome.workflow_persisted
            self.status = "completed"
            self.pending_requests = []
        except asyncio.CancelledError:
            self.status = "cancelled"
            self._sync_status()
            raise
        except ApprovalPendingError as exc:
            # Suspension, not failure: keep the pending requests for the inbox
            # and wait for a decision + resume(). No error is recorded — a plan
            # that later resumes and succeeds was never "failed".
            self.status = "waiting_approval"
            self._sync_status()
            self.pending_requests = list(exc.requests)
            _LOG.info(
                f"[plan-task {self.task_id}] waiting for approval: "
                f"{[request.intent for request in exc.requests]}"
            )
            self._notify_approvals()
        except Exception as exc:  # surface as task status, never crash the loop
            self.status = "failed"
            self.error = exc
            _LOG.warning(f"[plan-task {self.task_id}] failed: {exc!r}")
            # A terminally-failed plan still materializes: the Agents tab shows
            # it (status failed) and a FailureAnalysis records what happened.
            # (ApprovalPendingError never reaches here — its branch above keeps
            # a suspension out of the failure records.)
            from .materialize import PlanFailure

            failure = PlanFailure(stage=None, error=repr(exc))
            try:
                self.record_outcome = await asyncio.to_thread(
                    lambda: materialize_plan_records(
                        run=self.run,
                        experiment=self.experiment,
                        workspace_root=self.workspace_root,
                        task_id=self.task_id,
                        draft=self.draft,
                        model=self.model,
                        failure=failure,
                    )
                )
            except Exception as record_exc:  # pragma: no cover — records never mask the failure
                _LOG.warning(f"[plan-task {self.task_id}] failure records failed: {record_exc!r}")

    def resume(self) -> None:
        """Re-drive the pipeline after a decision landed in the approval store.

        The stage ledger skips completed stages; the suspending gate resolves
        store-first, so a stored grant passes and the pipeline proceeds (a
        rejection re-suspends honestly — rejections never replay).

        Raises:
            RuntimeError: The task is not ``waiting_approval`` — resuming a
                running/completed/failed task is a caller defect.
        """
        if self.status != "waiting_approval":
            raise RuntimeError(
                f"plan task {self.task_id!r} is {self.status!r}, not waiting_approval "
                "— only a suspended task can resume"
            )
        if self._gateway is None:
            raise RuntimeError(f"plan task {self.task_id!r} has no gateway to resume with")
        self.status = "running"
        self._sync_status()
        self.pending_requests = []
        self._task = asyncio.create_task(self._drive(self._gateway))

    def mark_rejected(self, reason: str) -> None:
        """Record an operator rejection: the current attempt ends ``failed``.

        The rejection itself is already persisted (approval store + event
        log) by the decide route; this flips the task's coarse status so the
        UI stops listing it as waiting. A later re-issue of the same draft
        re-asks (rejections never replay).
        """
        if self.status != "waiting_approval":
            raise RuntimeError(
                f"plan task {self.task_id!r} is {self.status!r}, not waiting_approval"
            )
        self.status = "failed"
        self._sync_status()
        self.error = RuntimeError(f"approval rejected: {reason}")
        self.pending_requests = []
        self._notify_approvals()

    def _sync_status(self) -> None:
        """Mirror the coarse status into the agent-task store (hub visibility)."""
        if not self.workspace_root:
            return
        from .record import write_plan_task_status

        write_plan_task_status(
            self.workspace_root,
            task_id=self.task_id,
            draft=self.draft,
            created_at=self.created_at,
            status=self.status,
        )

    @staticmethod
    def _notify_approvals() -> None:
        from molexp.services.approval_notify import notify_approvals_changed

        notify_approvals_changed()

    @property
    def run_id(self) -> str:
        return self.run.id

    def cancel(self) -> None:
        """Request cancellation of the background run (idempotent)."""
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def await_finished(self) -> None:
        """Await the background run, suppressing the cancellation it may raise."""
        if self._task is None:
            return
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
