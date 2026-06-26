"""``CurateTask`` — one background ``run_curation_flow`` run for the server.

Mirrors ``server/plan_runtime``'s ``PlanTask`` + registry: a curate task is
one-shot (run the shared :func:`~molexp.server.curate_runtime.flow.run_curation_flow`
once on a content-addressed Run), so the task IS the background ``asyncio.Task``
plus its coarse status. Approvals are auto-granted (the gate's default approver)
— interactive HTTP approval is a later concern.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Literal

from mollog import get_logger

if TYPE_CHECKING:
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.server.curate_runtime.flow import CurationResult
    from molexp.workspace import Experiment, Run, Workspace

__all__ = ["CurateTask", "CurateTaskRegistry", "CurateTaskStatus"]

_LOG = get_logger(__name__)

CurateTaskStatus = Literal["running", "completed", "failed", "cancelled"]


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
        task._task = asyncio.create_task(task._drive(gateway))
        return task

    async def _drive(self, gateway: AgentGateway) -> None:
        from molexp.server.curate_runtime.flow import run_curation_flow

        try:
            # approve=None → the gate auto-grants (non-interactive, like PlanTask).
            self.result = await run_curation_flow(
                self.request,
                workspace=self.workspace,
                experiment=self.experiment,
                run=self.run,
                gateway=gateway,
                approve=None,
            )
            self.status = "completed"
        except asyncio.CancelledError:
            self.status = "cancelled"
            raise
        except Exception as exc:  # surface as task status, never crash the loop
            self.status = "failed"
            self.error = exc
            _LOG.warning(f"[curate-task {self.task_id}] failed: {exc!r}")

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

    async def aclose(self) -> None:
        """Cancel and await every tracked task (server shutdown)."""
        for tasks in self._by_workspace.values():
            for task in tasks.values():
                task.cancel()
        for tasks in self._by_workspace.values():
            for task in tasks.values():
                await task.await_finished()
        self._by_workspace.clear()
