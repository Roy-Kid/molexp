"""Headline benchmark for workflow-workspace-hardening P1-8 (ac-012).

Measures the hot-path syscall cost of ``Caching.put`` against a *full* cache
holding ``E`` entries, for growing ``E``.

The legacy ``_evict_if_needed`` re-globbed the whole store and ``stat``-ed
every key on each put, so a single put cost ~O(E) syscalls and filling the
cache was quadratic. The in-process ``OrderedDict`` LRU evicts in O(1)
amortised — a steady-state put touches neither ``keys()`` (glob) nor
``access_time()`` (stat). This script reports the glob + stat call count per
put at each fill level so the before/after curve is directly comparable. It is
a MEASUREMENT harness, not a test: it never asserts, it only prints.

Run::

    python -m benches.bench_cache_put
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path

from molexp.workflow.cache import Caching
from molexp.workflow.cache_store import FileCacheStore
from molexp.workflow.snapshot import TaskSnapshot

# Cache fill levels (== max_entries, so the cache is full and every probe put
# triggers exactly one eviction) to probe.
FILL_LEVELS = (50, 100, 250, 500, 1000, 2000)
# Puts measured (each evicting one entry) at each fill level.
SAMPLE = 200


class _CountingStore:
    """CacheStore proxy counting the two O(E) hot-path operations."""

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


def _snapshot() -> TaskSnapshot:
    def body(_ctx: object) -> int:
        return 1

    return TaskSnapshot.from_task_body("t", body)


def main() -> None:
    snap = _snapshot()
    print(f"{'fill E':>8} {'glob/put':>12} {'stat/put':>12}")
    print("-" * 36)
    for level in FILL_LEVELS:
        with tempfile.TemporaryDirectory() as tmp:
            store = _CountingStore(FileCacheStore(Path(tmp) / "c"))
            cache = Caching(store=store, max_entries=level)
            for i in range(level):  # fill to capacity
                cache.put(snap, {"i": i}, {"r": i})
            store.keys_calls = 0
            store.access_time_calls = 0
            for i in range(level, level + SAMPLE):  # timed, each evicts one
                cache.put(snap, {"i": i}, {"r": i})
            print(
                f"{level:>8} {store.keys_calls / SAMPLE:>12.2f} "
                f"{store.access_time_calls / SAMPLE:>12.2f}"
            )
    print(
        "\nOrderedDict LRU keeps glob/put and stat/put at ~0 as E grows;\n"
        "the legacy glob+stat-per-key path grew linearly with E."
    )


if __name__ == "__main__":
    main()
