"""Structured trace sink.

Protocol + JSONL implementation. The orchestration layer emits one
trace record per harness event so post-hoc replay is trivial.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from molexp.agent.types import utc_now


class TraceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=utc_now)


@runtime_checkable
class TraceSink(Protocol):
    """Append-only structured-trace destination."""

    async def write(self, record: TraceRecord) -> None: ...


class NoopTraceSink:
    async def write(self, record: TraceRecord) -> None:
        return None


class JsonlTraceSink:
    """Append-only JSONL trace sink."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def write(self, record: TraceRecord) -> None:
        line = json.dumps(
            {
                "kind": record.kind,
                "payload": record.payload,
                "ts": record.ts.isoformat(),
            },
            ensure_ascii=False,
        )
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
