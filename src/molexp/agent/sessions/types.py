"""Session metadata type — the latest summary persisted to ``session.json``."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from molexp.agent.types import Goal, SessionStatus, utc_now


@dataclass(frozen=True)
class SessionMetadata:
    """Latest summary persisted to ``session.json``."""

    session_id: str
    goal: Goal
    status: SessionStatus
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    summary: str = ""


__all__ = ["SessionMetadata"]
