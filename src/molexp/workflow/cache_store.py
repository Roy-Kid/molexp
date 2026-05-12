"""Pluggable backing-storage for the workflow execution cache.

``Caching`` (in ``workflow.cache``) holds the cache-key derivation, format
versioning, and LRU eviction policy. The actual *storage* — read / write /
list / remove / atime — is delegated through the :class:`CacheStore`
Protocol so the cache can sit on top of either a plain filesystem
directory (``FileCacheStore``) or the workspace's singleton ``CacheFolder``
via ``ws.cache.as_cache_store()`` (returns a ``CacheStore``-conforming
adapter — see ``molexp.workspace.cache.folder``).

Sub-spec ``unify-folder-abstraction-03-system-folder-migration``
retired the standalone ``WorkspaceCacheStore`` class + the
``workflow.cache`` subsystem-kind string in favour of the
``CacheFolder.as_cache_store()`` adapter.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class CacheStore(Protocol):
    """String-keyed string-valued blob store with mtime-based LRU support.

    Implementations supply read / write / remove / list / atime; the
    cache-policy layer (:class:`Caching`) computes keys, encodes JSON,
    and decides when to evict.
    """

    def read(self, key: str) -> str | None:
        """Return the stored content for *key*, or ``None`` on miss."""

    def write(self, key: str, content: str) -> None:
        """Atomically write *content* under *key*."""

    def remove(self, key: str) -> bool:
        """Drop *key*. Returns ``True`` iff it was present."""

    def keys(self) -> Iterable[str]:
        """Yield every key currently stored."""

    def access_time(self, key: str) -> float:
        """Unix-timestamp of last access (for LRU); ``0.0`` if absent."""

    def touch(self, key: str) -> None:
        """Mark *key* as recently accessed (used on cache hits)."""

    def total_bytes(self) -> int:
        """Total bytes used by every stored entry."""

    def clear(self) -> int:
        """Drop every entry. Returns the number removed."""


# ── File-system implementation ──────────────────────────────────────────


def _entry_path(store_dir: Path, key: str) -> Path:
    return store_dir / f"{key}.json"


class FileCacheStore:
    """Filesystem-backed :class:`CacheStore` rooted at ``store_dir``.

    Each entry is one ``<key>.json`` file under *store_dir*. Atomic
    writes use a temp-file + rename (mirrors workspace's
    :func:`atomic_write_json` semantics for non-JSON-decoded strings).

    This is the right backing when a caller wants a cache that lives
    outside any workspace — e.g. the FastAPI server's process-local
    cache; library users running ad-hoc workflows; etc. Workspace-
    aware callers should reach for ``ws.cache.as_cache_store()``
    instead (returns a ``CacheStore`` adapter rooted at
    ``<workspace_root>/cache/``).
    """

    def __init__(self, store_dir: Path | str) -> None:
        self._store_dir = Path(store_dir)

    @property
    def store_dir(self) -> Path:
        return self._store_dir

    def _ensure_dir(self) -> None:
        self._store_dir.mkdir(parents=True, exist_ok=True)

    def read(self, key: str) -> str | None:
        path = _entry_path(self._store_dir, key)
        if not path.exists():
            return None
        try:
            return path.read_text()
        except OSError:
            return None

    def write(self, key: str, content: str) -> None:
        self._ensure_dir()
        path = _entry_path(self._store_dir, key)
        fd, tmp_str = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=f".{path.stem}_")
        tmp_path = Path(tmp_str)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            tmp_path.replace(path)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            raise

    def remove(self, key: str) -> bool:
        path = _entry_path(self._store_dir, key)
        if not path.exists():
            return False
        path.unlink(missing_ok=True)
        return True

    def keys(self) -> Iterable[str]:
        if not self._store_dir.exists():
            return ()
        return tuple(p.stem for p in self._store_dir.glob("*.json"))

    def access_time(self, key: str) -> float:
        path = _entry_path(self._store_dir, key)
        if not path.exists():
            return 0.0
        return path.stat().st_mtime

    def touch(self, key: str) -> None:
        path = _entry_path(self._store_dir, key)
        if path.exists():
            path.touch()

    def total_bytes(self) -> int:
        if not self._store_dir.exists():
            return 0
        return sum(p.stat().st_size for p in self._store_dir.glob("*.json"))

    def clear(self) -> int:
        if not self._store_dir.exists():
            return 0
        count = 0
        for p in self._store_dir.glob("*.json"):
            p.unlink(missing_ok=True)
            count += 1
        return count


__all__ = [
    "CacheStore",
    "FileCacheStore",
]
