"""``CurateTask`` — one background ``run_curation_flow`` run for the server.

Mirrors ``server/plan_runtime``'s ``PlanTask`` + registry: a curate task is
one-shot (run the shared :func:`~molexp.services.curate_runtime.flow.run_curation_flow`
once on a content-addressed Run), so the task IS the background ``asyncio.Task``
plus its coarse status.

Approvals are **never** auto-granted: the flow runs with ``approve=None``, so a
destructive op's ChangeProposal gate resolves store-first and otherwise
suspends — the task lands ``waiting_approval``, the approvals inbox lists the
pending request, and a granted decision re-drives the flow via
:meth:`CurateTask.resume`. Re-driving re-plans, but the proposal id is
content-derived, so the same request resolves against the stored grant and the
mutation executes; a *different* re-planned proposal re-suspends honestly.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Literal

from mollog import get_logger

if TYPE_CHECKING:
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.schemas import ApprovalRequest
    from molexp.services.curate_runtime.flow import CurationResult
    from molexp.workspace import Experiment, Run, Workspace

__all__ = ["CurateTask", "CurateTaskRegistry", "CurateTaskStatus"]

_LOG = get_logger(__name__)

CurateTaskStatus = Literal["running", "completed", "failed", "cancelled", "waiting_approval"]


class CurateTask:
    """A single background curation-flow run, its status, and its result."""

    def __init__(
        self,
        *,
        task_id: str,
        run: Run,
        experiment: Experiment,
        workspace: Workspace,
        request: str,
        model: str,
        created_at: str,
    ) -> None:
        self.task_id = task_id
        self.run = run
        self.experiment = experiment
        self.workspace = workspace
        self.request = request
        self.model = model
        self.created_at = created_at
        self.status: CurateTaskStatus = "running"
        self.error: BaseException | None = None
        self.result: CurationResult | None = None
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
        workspace: Workspace,
        request: str,
        model: str,
        created_at: str,
        gateway: AgentGateway,
    ) -> CurateTask:
        """Build a task and spawn its background curation-flow run."""
        task = cls(
            task_id=task_id,
            run=run,
            experiment=experiment,
            workspace=workspace,
            request=request,
            model=model,
            created_at=created_at,
        )
        task._gateway = gateway
        task._task = asyncio.create_task(task._drive(gateway))
        return task

    async def _drive(self, gateway: AgentGateway) -> None:
        from molexp.harness import ApprovalPendingError
        from molexp.services.curate_runtime.flow import run_curation_flow

        try:
            # approve=None → store-first: a stored grant executes, otherwise a
            # destructive op suspends pending (never auto-grants).
            self.result = await run_curation_flow(
                self.request,
                workspace=self.workspace,
                experiment=self.experiment,
                run=self.run,
                gateway=gateway,
                approve=None,
            )
            self.status = "completed"
            self.pending_requests = []
        except asyncio.CancelledError:
            self.status = "cancelled"
            raise
        except ApprovalPendingError as exc:
            # Suspension, not failure: the inbox lists the pending request and
            # a granted decision re-drives via resume().
            self.status = "waiting_approval"
            self.pending_requests = list(exc.requests)
            _LOG.info(
                f"[curate-task {self.task_id}] waiting for approval: "
                f"{[request.intent for request in exc.requests]}"
            )
            self._notify_approvals()
        except Exception as exc:  # surface as task status, never crash the loop
            self.status = "failed"
            self.error = exc
            _LOG.warning(f"[curate-task {self.task_id}] failed: {exc!r}")

    def resume(self) -> None:
        """Re-drive the flow after a decision landed in the approval store.

        Raises:
            RuntimeError: The task is not ``waiting_approval``.
        """
        if self.status != "waiting_approval":
            raise RuntimeError(
                f"curate task {self.task_id!r} is {self.status!r}, not waiting_approval "
                "— only a suspended task can resume"
            )
        if self._gateway is None:
            raise RuntimeError(f"curate task {self.task_id!r} has no gateway to resume with")
        self.status = "running"
        self.pending_requests = []
        self._task = asyncio.create_task(self._drive(self._gateway))

    def mark_rejected(self, reason: str) -> None:
        """Record an operator rejection: the current attempt ends ``failed``."""
        if self.status != "waiting_approval":
            raise RuntimeError(
                f"curate task {self.task_id!r} is {self.status!r}, not waiting_approval"
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


class CurateTaskRegistry:
    """Process-singleton store of background ``CurateTask``s, keyed by workspace."""

    def __init__(self) -> None:
        self._by_workspace: dict[str, dict[str, CurateTask]] = {}

    def create(
        self,
        *,
        workspace_root: str,
        task_id: str,
        run: Run,
        experiment: Experiment,
        workspace: Workspace,
        request: str,
        model: str,
        created_at: str,
        gateway: AgentGateway,
    ) -> CurateTask:
        """Spawn a ``CurateTask`` and store it under ``(workspace_root, task_id)``."""
        task = CurateTask.start(
            task_id=task_id,
            run=run,
            experiment=experiment,
            workspace=workspace,
            request=request,
            model=model,
            created_at=created_at,
            gateway=gateway,
        )
        self._by_workspace.setdefault(workspace_root, {})[task_id] = task
        return task

    def get(self, workspace_root: str, task_id: str) -> CurateTask | None:
        return self._by_workspace.get(workspace_root, {}).get(task_id)

    def list_tasks(self, workspace_root: str) -> list[CurateTask]:
        return list(self._by_workspace.get(workspace_root, {}).values())

    def pending_approvals(self, workspace_root: str) -> list[CurateTask]:
        """Return every ``waiting_approval`` task under ``workspace_root``."""
        return [
            task for task in self.list_tasks(workspace_root) if task.status == "waiting_approval"
        ]

    async def aclose(self) -> None:
        """Cancel and await every tracked task (server shutdown)."""
        for tasks in self._by_workspace.values():
            for task in tasks.values():
                task.cancel()
        for tasks in self._by_workspace.values():
            for task in tasks.values():
                await task.await_finished()
        self._by_workspace.clear()
