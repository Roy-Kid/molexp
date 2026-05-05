"""Memory record type — one entry in the harness memory log."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from molexp.agent.types import utc_now


@dataclass(frozen=True)
class MemoryRecord:
    """One stored memory entry."""

    kind: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=utc_now)


__all__ = ["MemoryRecord"]
