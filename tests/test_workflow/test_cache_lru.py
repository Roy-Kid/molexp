"""Caching LRU eviction cost (workflow-workspace-hardening P1-8 / ac-012).

``Caching.put`` evicts when the entry count exceeds ``max_entries``. The old
``_evict_if_needed`` re-globbed the whole store (``keys()``) and ``stat``-ed
every key (``access_time``) on *every* put, so a put against an E-entry cache
cost O(E) syscalls — quadratic fill. The in-process ``OrderedDict`` LRU makes
eviction O(1) amortised: a steady-state put touches neither ``keys()`` nor
``access_time()``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from molexp.workflow.cache import Caching
from molexp.workflow.cache_store import FileCacheStore
from molexp.workflow.snapshot import TaskSnapshot


def _snapshot() -> TaskSnapshot:
    def body(_ctx: object) -> int:
        return 1

    return TaskSnapshot.from_task_body("t", body)


class _CountingStore:
    """CacheStore proxy that counts the O(E) hot-path calls."""

    def __init__(self, inner: FileCacheStore) -> None:
        self._inner = inner
        self.access_time_calls = 0
        self.keys_calls = 0

    def read(self, key: str) -> str | None:
        return self._inner.read(key)

    def write(self, key: str, content: str) -> None:
        self._inner.write(key, content)

    def remove(self, key: str) -> bool:
        return self._inner.remove(key)

    def keys(self) -> Iterable[str]:
        self.keys_calls += 1
        return self._inner.keys()

    def access_time(self, key: str) -> float:
        self.access_time_calls += 1
        return self._inner.access_time(key)

    def touch(self, key: str) -> None:
        self._inner.touch(key)

    def total_bytes(self) -> int:
        return self._inner.total_bytes()

    def clear(self) -> int:
        return self._inner.clear()


def test_eviction_put_does_not_scan_or_stat(tmp_path: Path) -> None:
    """Steady-state puts against a full cache make zero ``keys()`` / O(E)
    ``access_time()`` calls (the old glob+stat-per-key path)."""
    store = _CountingStore(FileCacheStore(tmp_path / "c"))
    snap = _snapshot()
    max_entries = 50
    cache = Caching(store=store, max_entries=max_entries)

    # Fill to capacity.
    for i in range(max_entries):
        cache.put(snap, {"i": i}, {"r": i})

    # Reset counters; subsequent puts each evict exactly one entry.
    store.access_time_calls = 0
    store.keys_calls = 0

    puts = 100
    for i in range(max_entries, max_entries + puts):
        cache.put(snap, {"i": i}, {"r": i})

    # O(1) eviction: no full-directory glob, no per-key stat on the hot path.
    assert store.keys_calls == 0
    assert store.access_time_calls == 0


def test_lru_evicts_least_recently_used(tmp_path: Path) -> None:
    """Eviction order is LRU: a ``get`` refreshes recency so the touched
    entry survives while an untouched older one is evicted."""
    cache = Caching(store_dir=tmp_path / "c", max_entries=2)
    snap = _snapshot()

    cache.put(snap, {"i": 0}, {"r": 0})
    cache.put(snap, {"i": 1}, {"r": 1})
    # Touch entry 0 so it becomes most-recently-used.
    assert cache.get(snap, {"i": 0}) == {"r": 0}
    # Inserting entry 2 must evict entry 1 (the LRU), not entry 0.
    cache.put(snap, {"i": 2}, {"r": 2})

    assert cache.get(snap, {"i": 0}) == {"r": 0}
    assert cache.get(snap, {"i": 1}) is None  # evicted
    assert cache.get(snap, {"i": 2}) == {"r": 2}


def test_entry_count_bounded_by_max_entries(tmp_path: Path) -> None:
    cache = Caching(store_dir=tmp_path / "c", max_entries=10)
    snap = _snapshot()
    for i in range(100):
        cache.put(snap, {"i": i}, {"r": i})
    assert cache.stats["entry_count"] == 10
