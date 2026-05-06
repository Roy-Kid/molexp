"""Memory record type — one entry in the harness memory log."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from molexp.agent.types import utc_now


class MemoryRecord(BaseModel):
    """One stored memory entry."""

    model_config = ConfigDict(frozen=True)

    kind: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=utc_now)


__all__ = ["MemoryRecord"]
