"""``ModeResult`` — the frozen outcome of one :class:`molexp.harness.mode.Mode` run.

A pure-data record (no live services) carrying everything a caller needs after
a mode's stage pipeline has executed on the workflow engine: which mode ran,
the run + execution ids, the per-stage artifacts, the final artifact, and any
``PlanValidationReport``s produced by validator stages.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from molexp.harness.schemas.artifact import PlanArtifactRef
from molexp.harness.schemas.validation import PlanValidationReport

__all__ = ["ModeResult", "StageTiming"]


class StageTiming(BaseModel):
    """Wall-clock timing for one stage in a mode run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: str
    duration_s: float
    status: str = "ok"  # ok | skipped | failed


class ModeResult(BaseModel):
    """Outcome of one :class:`~molexp.harness.mode.Mode` run.

    Attributes:
        mode_name: The ``name`` of the mode that produced this result.
        run_id: The workspace ``Run`` id the pipeline executed under.
        execution_id: The workflow execution id (one attempt of the pipeline).
        stage_artifacts: Each stage's produced :class:`PlanArtifactRef`, in stage
            (topological) order.
        final_artifact: The terminal stage's artifact, or ``None`` if the
            pipeline produced no terminal artifact.
        validation_reports: Any :class:`PlanValidationReport`s emitted by validator
            stages during the run (empty when no validator ran or all passed
            without persisting a report).
        stage_timings: Per-stage wall times (including ledger skips).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode_name: str
    run_id: str
    execution_id: str
    stage_artifacts: tuple[PlanArtifactRef, ...] = ()
    final_artifact: PlanArtifactRef | None = None
    validation_reports: tuple[PlanValidationReport, ...] = ()
    stage_timings: tuple[StageTiming, ...] = Field(default_factory=tuple)
