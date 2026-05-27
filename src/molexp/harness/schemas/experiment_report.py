"""``ExperimentReport`` — the human-readable canonical experimental design.

Per ``.claude/notes/harness-goal.md`` §4.5: extends the user's informal
plan into a structured report that the rest of the pipeline (workflow IR
extraction, capability binding, test-spec generation) reads from. Required
fields are the four narrative anchors — title, objective, system, design;
everything else defaults to empty so an LLM proposal can omit fields it
doesn't know without failing validation.

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
