"""``PlanTaskRegistry`` — in-process registry of live PlanMode tasks.

Mirrors :class:`~molexp.server.agent_runtime.registry.AgentSessionRegistry`:
a process-singleton keyed ``(workspace_root, task_id)``, reached via the
``get_plan_runtime`` accessor (not ``app.state``) and reset on lifespan
shutdown. Tasks are in-memory only for the MVP — a server restart drops them,
but the run dir's content-addressed ledger means re-issuing the same draft
resumes the pipeline rather than starting over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.server.plan_runtime.task import PlanTask
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.run import Run

__all__ = ["PlanTaskRegistry"]


class PlanTaskRegistry:
    """Live :class:`PlanTask`s per workspace, keyed ``(workspace_root, task_id)``."""

    def __init__(self) -> None:
        self._by_workspace: dict[str, dict[str, PlanTask]] = {}

    def create(
        self,
        *,
        workspace_root: str,
        task_id: str,
        run: Run,
        experiment: Experiment,
        draft: str,
        model: str,
        created_at: str,
        gateway: AgentGateway,
    ) -> PlanTask:
        """Build, start, and register a background plan task."""
        from molexp.server.plan_runtime.task import PlanTask

        task = PlanTask.start(
            task_id=task_id,
            run=run,
            experiment=experiment,
            draft=draft,
            model=model,
            created_at=created_at,
            gateway=gateway,
        )
        self._by_workspace.setdefault(workspace_root, {})[task_id] = task
        return task

    def get(self, workspace_root: str, task_id: str) -> PlanTask | None:
        """Return the live task for ``(workspace_root, task_id)`` or ``None``."""
        return self._by_workspace.get(workspace_root, {}).get(task_id)

    def list_tasks(self, workspace_root: str) -> list[PlanTask]:
        """Return every live task under ``workspace_root``."""
        return list(self._by_workspace.get(workspace_root, {}).values())

    async def aclose(self) -> None:
        """Cancel and await every in-flight task — the lifespan shutdown hook."""
        tasks = [task for tasks in self._by_workspace.values() for task in tasks.values()]
        for task in tasks:
            task.cancel()
        for task in tasks:
            await task.await_finished()
        self._by_workspace.clear()
