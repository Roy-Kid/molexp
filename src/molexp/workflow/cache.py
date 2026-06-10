"""Execution result caching with LRU eviction and format versioning.

Each cache entry is keyed by the combination of a TaskSnapshot key and
an input hash::

    cache_key = sha256(snapshot.key | input_hash)

Features:

- LRU eviction when entry count exceeds ``max_entries``
- Format version field for forward-compatible schema evolution
- Pluggable backing storage via :class:`CacheStore`
  (``FileCacheStore`` for plain directories,
  ``ws.cache.as_cache_store()`` for workspace-rooted caches)

Usage::

    # Workspace-backed (recommended for in-process workflow runs):
    from molexp.workspace import Workspace
    from molexp.workflow import Caching

    ws = Workspace("./lab")
    cache = Caching(store=ws.cache.as_cache_store(), max_entries=1000)

    # Or plain filesystem (no workspace needed):
    from pathlib import Path
    from molexp.workflow import Caching

    cache = Caching(store_dir=Path("./cache"), max_entries=1000)

    hit = cache.get(snapshot, inputs)
    if hit is None:
        result = run_task(...)
        cache.put(snapshot, inputs, result)
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

from mollog import get_logger
from pydantic import BaseModel

from .._typing import HashablePayload, JSONValue
from .cache_store import CacheStore, FileCacheStore
from .snapshot import TaskSnapshot

logger = get_logger(__name__)

# Bump this when CacheEntry schema changes in a backwards-incompatible way.
CACHE_FORMAT_VERSION = 1

# Array-like inputs larger than this (element count) are content-hashed
# instead of fully serialized when computing cache keys, to avoid OOM on
# huge ndarrays.
LARGE_ARRAY_HASH_THRESHOLD_ELEMENTS = 10_000


def _robust_json_default(obj: HashablePayload) -> JSONValue:
    """Type-aware JSON serializer for cache input hashing.

    Large array-like objects (> :data:`LARGE_ARRAY_HASH_THRESHOLD_ELEMENTS`
    elements) are hashed rather than fully serialized to avoid excessive
    memory usage.
    """
    if hasattr(obj, "tolist"):
        size = obj.size if hasattr(obj, "size") else len(obj)
        if size > LARGE_ARRAY_HASH_THRESHOLD_ELEMENTS:
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
    result: dict[str, JSONValue]
    created_at: datetime


class Caching:
    """Execution cache with LRU eviction; storage delegated to a :class:`CacheStore`.

    Two construction shapes:

    - ``Caching(store=<CacheStore>)`` — supply any store directly. The
      preferred form for workspace-backed caches:
      ``Caching(store=ws.cache.as_cache_store())``.
    - ``Caching(store_dir=<Path>)`` — backward-compat shorthand that
      builds a :class:`FileCacheStore` rooted at *store_dir*.

    Args:
        store: Pluggable :class:`CacheStore` impl. Mutually exclusive
            with *store_dir*.
        store_dir: Filesystem directory for cache files. Mutually
            exclusive with *store*.
        max_entries: Maximum number of cache entries (0 = unlimited).

    Raises:
        ValueError: If both *store* and *store_dir* are passed (or
            neither).
    """

    def __init__(
        self,
        *,
        store: CacheStore | None = None,
        store_dir: Path | str | None = None,
        max_entries: int = 10000,
    ) -> None:
        if store is None and store_dir is None:
            raise ValueError("Caching requires exactly one of `store` or `store_dir`.")
        if store is not None and store_dir is not None:
            raise ValueError(
                "Caching does not accept both `store` and `store_dir`. Pass exactly one."
            )
        if store is not None:
            self._store: CacheStore = store
        else:
            assert store_dir is not None  # guarded by the checks above
            self._store = FileCacheStore(Path(store_dir))
        self._max_entries = max_entries
        # In-process LRU index: cache_key → None, ordered oldest-first. Built
        # lazily from the store on first use (one O(E) glob+stat) and then
        # maintained incrementally so eviction is O(1) amortised — no
        # per-put directory glob or per-key stat. ``None`` = not yet loaded.
        self._lru: OrderedDict[str, None] | None = None

    def _ensure_lru(self) -> OrderedDict[str, None]:
        """Lazily materialise the LRU index from the backing store.

        Done once per :class:`Caching` instance: the store may already hold
        entries from a previous process, ordered by on-disk access time.
        """
        if self._lru is None:
            keys = list(self._store.keys())
            keys.sort(key=self._store.access_time)  # oldest first → LRU front
            self._lru = OrderedDict((k, None) for k in keys)
        return self._lru

    def _mark_used(self, cache_key: str) -> None:
        """Record *cache_key* as most-recently-used (insert or move to end)."""
        lru = self._ensure_lru()
        lru[cache_key] = None
        lru.move_to_end(cache_key)

    def _drop(self, cache_key: str) -> None:
        """Forget *cache_key* from the LRU index (no store interaction)."""
        if self._lru is not None:
            self._lru.pop(cache_key, None)

    @property
    def store(self) -> CacheStore:
        return self._store

    @property
    def stats(self) -> dict[str, JSONValue]:
        """Cache statistics: entry_count, total_size_bytes, max_entries."""
        # ``self._store.keys()`` is a Protocol method, not a dict view —
        # the SIM118 lint that suggests dropping ``.keys()`` would break
        # the call (``self._store`` is not iterable directly).
        return {
            "entry_count": sum(1 for _ in self._store.keys()),  # noqa: SIM118
            "total_size_bytes": self._store.total_bytes(),
            "max_entries": self._max_entries,
        }

    def get(
        self, snapshot: TaskSnapshot, inputs: dict[str, JSONValue]
    ) -> dict[str, JSONValue] | None:
        """Look up a cached result. Returns None on miss, version mismatch, or corruption."""
        cache_key = self._compute_cache_key(snapshot.key, self._compute_input_hash(inputs))
        raw = self._store.read(cache_key)
        if raw is None:
            return None

        try:
            entry = CacheEntry.model_validate_json(raw)
        except Exception:
            logger.warning(f"Corrupted cache entry {cache_key}, removing")
            self._store.remove(cache_key)
            self._drop(cache_key)
            return None

        if entry.version != CACHE_FORMAT_VERSION:
            logger.debug(
                f"Cache version mismatch for {cache_key}: "
                f"entry={entry.version}, current={CACHE_FORMAT_VERSION}"
            )
            self._store.remove(cache_key)
            self._drop(cache_key)
            return None

        self._store.touch(cache_key)
        self._mark_used(cache_key)
        return entry.result

    def put(
        self,
        snapshot: TaskSnapshot,
        inputs: dict[str, JSONValue],
        result: dict[str, JSONValue],
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
            created_at=datetime.now(UTC),
        )
        self._store.write(cache_key, entry.model_dump_json(indent=2))
        if self._max_entries > 0:
            self._mark_used(cache_key)
            self._evict_if_needed()

    def invalidate(self, snapshot: TaskSnapshot, inputs: dict[str, JSONValue]) -> bool:
        """Remove a single cache entry. Returns True if the entry existed."""
        cache_key = self._compute_cache_key(snapshot.key, self._compute_input_hash(inputs))
        existed = self._store.remove(cache_key)
        self._drop(cache_key)
        return existed

    def clear(self) -> int:
        """Remove all cache entries. Returns count removed."""
        removed = self._store.clear()
        self._lru = OrderedDict()
        return removed

    def _evict_if_needed(self) -> None:
        """Evict least-recently-used entries until within ``max_entries``.

        O(1) amortised per eviction: pop the oldest key off the in-process
        ``OrderedDict`` and drop its file. No directory glob, no per-key stat.
        """
        lru = self._ensure_lru()
        while len(lru) > self._max_entries:
            old_key, _ = lru.popitem(last=False)  # oldest = LRU front
            self._store.remove(old_key)
            logger.debug(f"Evicted cache entry: {old_key}")

    @staticmethod
    def _compute_cache_key(snapshot_key: str, input_hash: str) -> str:
        combined = f"{snapshot_key}|{input_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()

    @staticmethod
    def _compute_input_hash(inputs: dict[str, JSONValue]) -> str:
        raw = json.dumps(inputs, sort_keys=True, default=_robust_json_default)
        return hashlib.sha256(raw.encode()).hexdigest()
