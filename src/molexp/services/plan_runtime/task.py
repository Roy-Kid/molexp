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
    from molexp.workspace.experiment import Experiment
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
    ) -> None:
        self.task_id = task_id
        self.run = run
        self.experiment = experiment
        self.draft = draft
        self.model = model
        self.created_at = created_at
        self.ground = ground
        self.workspace_root = workspace_root
        self.status: PlanTaskStatus = "running"
        self.error: BaseException | None = None
        self.workflow_persisted = False
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
    ) -> PlanTask:
        """Build a task and spawn its background PlanMode run."""
        task = cls(
            task_id=task_id,
            run=run,
            experiment=experiment,
            draft=draft,
            model=model,
            created_at=created_at,
            ground=ground,
            workspace_root=workspace_root,
        )
        task._gateway = gateway
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

                capability_registry = await aresolve_capability_registry(self.workspace_root)
            # drive_plan_mode wraps the pipeline in the run lifecycle so the
            # plan Run's status is honest (running -> succeeded | failed) —
            # the same shared path `molexp plan` uses.
            self.result = await drive_plan_mode(
                PlanMode(),
                run=self.run,
                user_input=self.draft,
                gateway=gateway,
                capability_registry=capability_registry,
            )
            # Persist the workflow IR + record the Agents-tab session and
            # Knowledge note. Shared with `molexp plan` (CLI) so the Python and
            # UI paths land identical workspace state. Blocking I/O — offloaded.
            self.workflow_persisted = await asyncio.to_thread(
                lambda: materialize_plan_records(
                    run=self.run,
                    experiment=self.experiment,
                    workspace_root=self.workspace_root,
                    task_id=self.task_id,
                    draft=self.draft,
                    model=self.model,
                )
            )
            self.status = "completed"
            self.pending_requests = []
        except asyncio.CancelledError:
            self.status = "cancelled"
            raise
        except ApprovalPendingError as exc:
            # Suspension, not failure: keep the pending requests for the inbox
            # and wait for a decision + resume(). No error is recorded — a plan
            # that later resumes and succeeds was never "failed".
            self.status = "waiting_approval"
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
        self.error = RuntimeError(f"approval rejected: {reason}")
        self.pending_requests = []
        self._notify_approvals()

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
