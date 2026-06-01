"""``UserPlan`` — the verbatim user input plus structured envelope.

Per ``.claude/notes/harness-goal.md`` §4.4: the user's original natural-
language plan is the audit anchor for every downstream artifact. The
``UserPlan`` model wraps the raw text with optional identity / submission
metadata; the verbatim text itself lives as a separate ``user_plan``-kind
artifact (written by :class:`molexp.harness.stages.save_user_plan.SaveUserPlan`)
that the ``UserPlan`` JSON references through provenance edges.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from molexp.harness.schemas.artifact import ArtifactRef

__all__ = ["UserPlan"]


class UserPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw_text: str
    user_id: str | None = None
    submitted_at: datetime
    attachments: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
