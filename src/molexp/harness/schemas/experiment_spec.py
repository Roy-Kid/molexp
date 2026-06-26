"""``ExperimentSpec`` — the concrete, parameter-resolved specification.

Step 2 of the plan pipeline. The :class:`ExperimentReport` is a
human-readable proposal whose ``variables`` / ``controlled_conditions`` are
free-text strings and whose ``user_questions`` are open. ``ExperimentSpec``
is the **concrete** layer below it: every variable and condition is lifted
into a structured, provenance-carrying :class:`ParameterValue`, and every
open question is given a resolved answer. It is what the rest of the
pipeline (IR extraction, input-set expansion) reads from once the design is
pinned down.

Persisted twice by ``GenerateExperimentSpec`` — once as JSON (the machine
artifact downstream stages parse) and once as a human-readable YAML
companion under the same ``experiment_spec`` kind.

Frozen pydantic so a downstream stage cannot mutate the pinned spec.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from molexp.harness.schemas.parameter import ParameterSource, ParameterValue

__all__ = [
    "ExperimentSpec",
    "ResolvedQuestion",
    "SpecCondition",
    "SpecVariable",
]


class SpecVariable(BaseModel):
    """A concretized experimental variable lifted from the report's free text."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: ParameterValue
    unit: str | None = None
    description: str | None = None
    expected_type: str | None = None


class SpecCondition(BaseModel):
    """A concretized controlled condition (held fixed across the experiment)."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: ParameterValue
    unit: str | None = None
    notes: str | None = None


class ResolvedQuestion(BaseModel):
    """An answer to one of the report's open ``user_questions``."""

    model_config = ConfigDict(frozen=True)

    question: str
    answer: str
    source: ParameterSource = "agent_inferred"
    confidence: float | None = None


class ExperimentSpec(BaseModel):
    """Concrete specification derived from a human-readable experiment report."""

    model_config = ConfigDict(frozen=True)

    id: str
    experiment_report_id: str
    title: str
    objective: str
    variables: list[SpecVariable] = Field(default_factory=list)
    controlled_conditions: list[SpecCondition] = Field(default_factory=list)
    resolved_questions: list[ResolvedQuestion] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
