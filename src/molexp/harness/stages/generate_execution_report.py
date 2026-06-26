"""``GenerateExecutionReport`` — plan step 9: the "where & how" hand-off.

Pure stage (no LLM): synthesizes an :class:`ExecutionReport` from the bound
workflow (``execution_backend`` + :class:`ResourcePolicy` +
:class:`ExecutionEnvironment`), the latest ``input_set`` (``total_runs``),
and the workspace :class:`ComputeTarget` chosen for the run (injected — the
harness ``ctx`` exposes only the run dir, not the workspace target registry).

Descriptive only: it answers "which machine, which account, how many runs,
under what limits" for the step-8 reviewer. It never submits anything; real
execution is the explicit ``--execute`` tail.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.schemas import ArtifactRef, BoundWorkflow, ExecutionReport, InputSet
from molexp.harness.stages._resolve import require_latest
from molexp.workspace.utils import generate_id

if TYPE_CHECKING:
    from molexp.workspace.models import ComputeTarget

__all__ = ["GenerateExecutionReport"]


class GenerateExecutionReport(Stage):
    """Synthesize the descriptive ExecutionReport artifact (no submission)."""

    name: ClassVar[str] = "generate_execution_report"

    def __init__(self, compute_target: ComputeTarget | None = None) -> None:
        self._target = compute_target

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        bw_ref = require_latest(ctx, "bound_workflow", stage=self.name)
        try:
            bound = BoundWorkflow.model_validate_json(ctx.artifact_store.get(bw_ref.id))
        except Exception as exc:
            raise StageExecutionError(
                f"stage {self.name!r} could not parse the 'bound_workflow' artifact "
                f"{bw_ref.id!r}: {exc!r}"
            ) from exc

        total_runs = self._total_runs(ctx)
        target = self._target
        scheduling = dict(getattr(target, "default_scheduling", {}) or {}) if target else {}

        report = ExecutionReport(
            id=f"execution-report-{generate_id()}",
            bound_workflow_id=bound.id,
            target_name=getattr(target, "name", None) or "local",
            scheduler=getattr(target, "scheduler", None) or "local",
            host=getattr(target, "host", None),
            scratch_root=getattr(target, "scratch_root", None),
            account=_as_str(scheduling.get("account")),
            queue=_as_str(scheduling.get("queue")),
            partition=_as_str(scheduling.get("partition")),
            total_runs=total_runs,
            resource_policy=bound.resource_policy,
            environment=bound.environment,
            notes=list(bound.review_flags),
        )
        return ctx.artifact_store.put_json(
            kind="execution_report",
            obj=json.loads(report.model_dump_json()),
            created_by="GenerateExecutionReport",
            parent_ids=[bw_ref.id],
        )

    @staticmethod
    def _total_runs(ctx: HarnessRunContext) -> int:
        ref = ctx.artifact_store.latest_by_kind("input_set")
        if ref is None:
            return 1
        try:
            return InputSet.model_validate_json(ctx.artifact_store.get(ref.id)).total_runs
        except Exception:
            return 1


def _as_str(value: object) -> str | None:
    return str(value) if value is not None else None
