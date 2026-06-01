"""``ParameterValue`` + ``ParameterSource`` — provenance for every parameter.

Per ``.claude/notes/harness-goal.md`` §4.3 and §1.6: every scientifically
meaningful parameter that flows through the harness MUST carry its source
so the audit report can answer "where did this number come from?" without
guessing.

The seven-value ``ParameterSource`` set is the harness's contract with
agents: an agent proposing a NEMD ``field_strength = 1e6 V/cm`` MUST tag
it ``agent_inferred`` (which then trips the ``ApprovalPolicy`` in a later
Phase); a user-supplied value is ``user_provided``; a literature lookup is
``literature_default`` with a ``citation``; etc.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

__all__ = ["ParameterSource", "ParameterValue"]


ParameterSource = Literal[
    "user_provided",
    "agent_inferred",
    "project_default",
    "package_default",
    "literature_default",
    "manual_override",
    "runtime_detected",
]


class ParameterValue(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: Any
    source: ParameterSource
    reason: str | None = None
    confidence: float | None = None
    citation: str | None = None
    approved: bool = False
