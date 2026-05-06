"""Session metadata type — the latest summary persisted to ``session.json``."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from molexp.agent.types import Goal, SessionStatus, utc_now


class SessionMetadata(BaseModel):
    """Latest summary persisted to ``session.json``."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    goal: Goal
    status: SessionStatus
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    summary: str = ""


__all__ = ["SessionMetadata"]
