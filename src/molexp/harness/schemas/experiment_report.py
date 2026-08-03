"""``ExperimentReport`` — the human-readable canonical experimental design.

Per ``.claude/notes/harness-goal.md`` §4.5: extends the user's informal
plan into a structured report that the rest of the pipeline (workflow IR
extraction, capability binding, test-spec generation) reads from. Required
fields are the four narrative anchors — title, objective, system, design;
everything else defaults to empty so an LLM proposal can omit fields it
doesn't know without failing validation.

The operator-facing **experiment plan book** is ``body_md`` — a full markdown
document following the 12-section outline in
:mod:`molexp.harness.plan.document` (Goal → Questions → Background →
Hypotheses → Design → Workflow → Tasks → Analysis → Success → Risks →
Outcomes → Deliverables → References). Structured fields remain for pipeline
consumers; the UI prefers ``body_md`` when present.

Frozen pydantic so a downstream stage that reads this report cannot mutate
it mid-pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ExperimentReport"]


class ExperimentReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    objective: str
    background: str | None = None
    system_description: str
    scientific_hypothesis: str | None = None
    experimental_design: str
    variables: list[str] = Field(default_factory=list)
    controlled_conditions: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks_or_uncertainties: list[str] = Field(default_factory=list)
    user_questions: list[str] = Field(default_factory=list)
    #: Full 12-section experiment plan book (markdown). Preferred for UI/readout.
    body_md: str | None = None

    def to_document_md(self) -> str:
        """Return the canonical plan book markdown (``body_md`` or reconstructed)."""
        from molexp.harness.plan.document import experiment_report_to_document

        return experiment_report_to_document(self)
