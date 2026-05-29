"""``ModeResult`` — the frozen outcome of one :class:`molexp.harness.mode.Mode` run.

A pure-data record (no live services) carrying everything a caller needs after
a mode's stage pipeline has executed on the workflow engine: which mode ran,
the run + execution ids, the per-stage artifacts, the final artifact, and any
``ValidationReport``s produced by validator stages.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from molexp.harness.schemas.artifact import ArtifactRef
from molexp.harness.schemas.validation import ValidationReport

__all__ = ["ModeResult"]


class ModeResult(BaseModel):
    """Outcome of one :class:`~molexp.harness.mode.Mode` run.

    Attributes:
        mode_name: The ``name`` of the mode that produced this result.
        run_id: The workspace ``Run`` id the pipeline executed under.
        execution_id: The workflow execution id (one attempt of the pipeline).
        stage_artifacts: Each stage's produced :class:`ArtifactRef`, in stage
            (topological) order.
        final_artifact: The terminal stage's artifact, or ``None`` if the
            pipeline produced no terminal artifact.
        validation_reports: Any :class:`ValidationReport`s emitted by validator
            stages during the run (empty when no validator ran or all passed
            without persisting a report).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode_name: str
    run_id: str
    execution_id: str
    stage_artifacts: tuple[ArtifactRef, ...] = ()
    final_artifact: ArtifactRef | None = None
    validation_reports: tuple[ValidationReport, ...] = ()
