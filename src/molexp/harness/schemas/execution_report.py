"""``ExecutionReport`` — the descriptive "where & how this will run" hand-off.

Step 9, the terminal artifact of the plan pipeline. It synthesizes — from
the :class:`BoundWorkflow` (``execution_backend`` + :class:`ResourcePolicy`
+ :class:`ExecutionEnvironment`) and the workspace ``ComputeTarget`` chosen
for the run — a single human-reviewable summary answering "which machine,
which account, how many runs, under what limits".

It is **descriptive only**. The harness never submits a job from this
report (north-star §2.2 forbids auto-submission); real execution is the
explicit opt-in ``--execute`` tail, gated by the step-8 review. The
machine/account fields mirror ``workspace.ComputeTarget`` without importing
it into the schema — the producing stage flattens the target it is handed.

Frozen pydantic.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from molexp.harness.schemas.bound_workflow import ExecutionEnvironment, ResourcePolicy

__all__ = ["ExecutionReport"]


class ExecutionReport(BaseModel):
    """Pre-execution summary: the resolved compute target plus run limits."""

    model_config = ConfigDict(frozen=True)

    id: str
    bound_workflow_id: str
    target_name: str
    scheduler: Literal["local", "slurm", "pbs", "lsf"] = "local"
    host: str | None = None
    scratch_root: str | None = None
    account: str | None = None
    queue: str | None = None
    partition: str | None = None
    total_runs: int = 1
    resource_policy: ResourcePolicy
    environment: ExecutionEnvironment
    notes: list[str] = []
