"""Memory store protocol and initial implementations.

Memory is an append-only stream — :class:`MemoryRecord` lines, no CRUD,
no three-tier shadowing. This is intentionally distinct from
:mod:`molexp.agent.persistence` which models named-resource CRUD; the
storage shape is wrong (JSONL append vs JSON-object map) so the
generic primitive does not apply.

Ships a no-op store plus a JSONL append-only stub. Embeddings and
vector indexes are out of scope for this revision.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from molexp.agent.memory.types import MemoryRecord


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
    """Append-only JSONL memory store."""

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
        # JSONL append is naturally crash-safe for whole-line writes;
        # we still fsync to keep semantics tight on shared filesystems.
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


__all__ = ["JsonlMemoryStore", "MemoryStore", "NoopMemoryStore"]
