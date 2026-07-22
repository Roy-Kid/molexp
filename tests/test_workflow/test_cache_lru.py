"""``Caching`` LRU eviction ordering (workflow-workspace-hardening P1-8).

``Caching.put`` evicts when the entry count exceeds ``max_entries``; the
victim is the least-recently-used entry, and a ``get`` refreshes recency.
"""

from __future__ import annotations

from pathlib import Path

from molexp.workflow.cache import Caching
from molexp.workflow.snapshot import TaskSnapshot


def _snapshot() -> TaskSnapshot:
    def body(_ctx: object) -> int:
        return 1

    return TaskSnapshot.from_task_body("t", body)


class TestCachingLRUEviction:
    def test_get_refreshes_recency_so_untouched_entry_is_evicted(self, tmp_path: Path) -> None:
        """A ``get`` marks its entry most-recently-used, so a later insert evicts
        the untouched older entry rather than the touched one."""
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
