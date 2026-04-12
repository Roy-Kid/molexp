"""Execution result caching with LRU eviction and format versioning.

Each cache entry is keyed by the combination of a TaskSnapshot key and an input hash:
    cache_key = f(snapshot.key, input_hash)

Features:
    - LRU eviction when entry count exceeds max_entries
    - Format version field for forward-compatible schema evolution
    - Atomic file writes to prevent corruption
    - Robust input hashing with type-aware serialization

Usage:
    cache = Caching(store_dir=Path("~/.molexp/cache"), max_entries=1000)
    cache.initialize()

    hit = cache.get(snapshot, inputs)
    if hit is None:
        result = run_task(...)
        cache.put(snapshot, inputs, result)
"""

from __future__ import annotations

import hashlib
import json
from mollog import get_logger
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .snapshot import TaskSnapshot

logger = get_logger(__name__)

# Bump this when CacheEntry schema changes in a backwards-incompatible way.
CACHE_FORMAT_VERSION = 1


def _robust_json_default(obj: Any) -> Any:
    """Type-aware JSON serializer for cache input hashing.

    Large array-like objects (>10 000 elements) are hashed rather than
    fully serialized to avoid excessive memory usage.
    """
    if hasattr(obj, "tolist"):
        size = obj.size if hasattr(obj, "size") else len(obj)
        if size > 10_000:
            # Hash large arrays instead of serializing to avoid OOM
            raw_bytes = obj.tobytes() if hasattr(obj, "tobytes") else str(obj).encode()
            digest = hashlib.sha256(raw_bytes).hexdigest()
            return {
                "__type__": "ndarray_digest",
                "shape": list(obj.shape) if hasattr(obj, "shape") else [size],
                "dtype": str(obj.dtype),
                "digest": digest,
            }
        return {"__type__": "ndarray", "data": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, Path):
        return {"__type__": "Path", "path": str(obj)}
    if isinstance(obj, datetime):
        return {"__type__": "datetime", "iso": obj.isoformat()}
    if hasattr(obj, "model_dump"):
        return {"__type__": "pydantic", "data": obj.model_dump()}
    return str(obj)


class CacheEntry(BaseModel):
    """A single cache record = snapshot + inputs + result."""

    version: int = CACHE_FORMAT_VERSION
    snapshot_key: str
    input_hash: str
    cache_key: str
    task_id: str
    task_type: str
    result: dict[str, Any]
    created_at: datetime


class Caching:
    """File-backed execution cache with LRU eviction.

    Each entry is keyed by the combination of a TaskSnapshot key and the hash of
    the concrete inputs.

    Args:
        store_dir: Directory for cache files.
        max_entries: Maximum number of cache entries (0 = unlimited).
    """

    def __init__(self, store_dir: Path, max_entries: int = 10000) -> None:
        self._store_dir = Path(store_dir)
        self._max_entries = max_entries

    def initialize(self) -> None:
        """Ensure the store directory exists."""
        self._store_dir.mkdir(parents=True, exist_ok=True)

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics: entry_count, total_size_bytes, max_entries."""
        if not self._store_dir.exists():
            return {"entry_count": 0, "total_size_bytes": 0, "max_entries": self._max_entries}
        entries = list(self._store_dir.glob("*.json"))
        total_size = sum(p.stat().st_size for p in entries)
        return {
            "entry_count": len(entries),
            "total_size_bytes": total_size,
            "max_entries": self._max_entries,
        }

    def get(self, snapshot: TaskSnapshot, inputs: dict[str, Any]) -> dict[str, Any] | None:
        """Look up a cached result. Returns None on miss, version mismatch, or corruption."""
        cache_key = self._compute_cache_key(snapshot.key, self._compute_input_hash(inputs))
        path = self._key_path(cache_key)
        if not path.exists():
            return None

        try:
            entry = CacheEntry.model_validate_json(path.read_text())
        except Exception:
            logger.warning(f"Corrupted cache entry {path.name}, removing")
            path.unlink(missing_ok=True)
            return None

        if entry.version != CACHE_FORMAT_VERSION:
            logger.debug(
                f"Cache version mismatch for {cache_key}: "
                f"entry={entry.version}, current={CACHE_FORMAT_VERSION}"
            )
            path.unlink(missing_ok=True)
            return None

        path.touch()
        return entry.result

    def put(
        self,
        snapshot: TaskSnapshot,
        inputs: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Store a result in the cache. Triggers LRU eviction if limit exceeded."""
        input_hash = self._compute_input_hash(inputs)
        cache_key = self._compute_cache_key(snapshot.key, input_hash)
        entry = CacheEntry(
            version=CACHE_FORMAT_VERSION,
            snapshot_key=snapshot.key,
            input_hash=input_hash,
            cache_key=cache_key,
            task_id=snapshot.task_id,
            task_type=snapshot.task_type,
            result=result,
            created_at=datetime.now(timezone.utc),
        )
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self._key_path(cache_key), entry.model_dump_json(indent=2))
        if self._max_entries > 0:
            self._evict_if_needed()

    def invalidate(self, snapshot: TaskSnapshot, inputs: dict[str, Any]) -> bool:
        """Remove a single cache entry. Returns True if the entry existed."""
        cache_key = self._compute_cache_key(snapshot.key, self._compute_input_hash(inputs))
        path = self._key_path(cache_key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Remove all cache entries. Returns count removed."""
        count = 0
        if self._store_dir.exists():
            for p in self._store_dir.glob("*.json"):
                p.unlink()
                count += 1
        return count

    def _key_path(self, cache_key: str) -> Path:
        return self._store_dir / f"{cache_key}.json"

    def _evict_if_needed(self) -> None:
        entries = list(self._store_dir.glob("*.json"))
        if len(entries) <= self._max_entries:
            return
        entries.sort(key=lambda p: p.stat().st_mtime)
        to_remove = len(entries) - self._max_entries
        for path in entries[:to_remove]:
            path.unlink(missing_ok=True)
            logger.debug(f"Evicted cache entry: {path.name}")

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, suffix=".tmp", prefix=f".{path.stem}_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _compute_cache_key(snapshot_key: str, input_hash: str) -> str:
        combined = f"{snapshot_key}|{input_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()

    @staticmethod
    def _compute_input_hash(inputs: dict[str, Any]) -> str:
        raw = json.dumps(inputs, sort_keys=True, default=_robust_json_default)
        return hashlib.sha256(raw.encode()).hexdigest()
