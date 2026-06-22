"""``PlanTask`` — one background PlanMode pipeline run for the server.

A plan task is one-shot (run PlanMode once on a content-addressed Run), so —
unlike the agent-session runtime — it needs no session/turn split: the task IS
the background ``asyncio.Task`` plus its coarse status. On success it persists
the generated workflow onto the experiment so the UI graph renderer shows it.
Approvals are auto-granted (``PlanMode()`` with no approver) — interactive
HTTP approval is a later concern.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Literal

from mollog import get_logger

if TYPE_CHECKING:
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.schemas import ModeResult
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.run import Run

__all__ = ["PlanTask", "PlanTaskStatus"]

_LOG = get_logger(__name__)

PlanTaskStatus = Literal["running", "completed", "failed", "cancelled"]


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
    ) -> None:
        self.task_id = task_id
        self.run = run
        self.experiment = experiment
        self.draft = draft
        self.model = model
        self.created_at = created_at
        self.status: PlanTaskStatus = "running"
        self.error: BaseException | None = None
        self.workflow_persisted = False
        self.result: ModeResult | None = None
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
    ) -> PlanTask:
        """Build a task and spawn its background PlanMode run."""
        task = cls(
            task_id=task_id,
            run=run,
            experiment=experiment,
            draft=draft,
            model=model,
            created_at=created_at,
        )
        task._task = asyncio.create_task(task._drive(gateway))
        return task

    async def _drive(self, gateway: AgentGateway) -> None:
        from molexp.harness import PlanMode

        from .persist import persist_plan_workflow_to_experiment

        try:
            self.result = await PlanMode().run(run=self.run, user_input=self.draft, gateway=gateway)
            # Compile + persist is blocking (exec + file write) — offload it.
            self.workflow_persisted = await asyncio.to_thread(
                persist_plan_workflow_to_experiment, self.run, self.experiment
            )
            self.status = "completed"
        except asyncio.CancelledError:
            self.status = "cancelled"
            raise
        except Exception as exc:  # surface as task status, never crash the loop
            self.status = "failed"
            self.error = exc
            _LOG.warning(f"[plan-task {self.task_id}] failed: {exc!r}")

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
