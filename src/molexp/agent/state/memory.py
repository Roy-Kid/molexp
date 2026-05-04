"""Memory store protocol + initial implementations (spec §6.4).

Phase 0/1a only ship the no-op store and the JSONL append-only stub.
No embeddings, no vector indexes — those land in a later spec.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from molexp.agent.types import utc_now


@dataclass(frozen=True)
class MemoryRecord:
    """One stored memory entry."""

    kind: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=utc_now)


@runtime_checkable
class MemoryStore(Protocol):
    """Read/write contract for harness memory."""

    async def append(self, record: MemoryRecord) -> None: ...

    async def list(self, kind: str | None = None) -> tuple[MemoryRecord, ...]: ...


class NoopMemoryStore:
    """Default store: drops writes, returns empty on reads."""

    async def append(self, record: MemoryRecord) -> None:
        return None

    async def list(self, kind: str | None = None) -> tuple[MemoryRecord, ...]:
        return ()


class JsonlMemoryStore:
    """Append-only JSONL memory store (spec §6.4)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def append(self, record: MemoryRecord) -> None:
        line = json.dumps(
            {
                "kind": record.kind,
                "content": record.content,
                "metadata": record.metadata,
                "ts": record.ts.isoformat(),
            }
        )
        # JSONL append is naturally crash-safe for whole-line writes; we
        # still fsync to keep semantics tight on shared filesystems.
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    async def list(self, kind: str | None = None) -> tuple[MemoryRecord, ...]:
        if not self._path.exists():
            return ()
        records: list[MemoryRecord] = []
        with self._path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                if kind is not None and payload.get("kind") != kind:
                    continue
                records.append(
                    MemoryRecord(
                        kind=payload["kind"],
                        content=payload["content"],
                        metadata=payload.get("metadata", {}),
                        ts=datetime.fromisoformat(payload["ts"]),
                    )
                )
        return tuple(records)
