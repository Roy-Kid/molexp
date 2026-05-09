"""Pluggable backing-storage for the workflow execution cache.

``Caching`` (in ``workflow.cache``) holds the cache-key derivation, format
versioning, and LRU eviction policy. The actual *storage* — read / write /
list / remove / atime — is delegated through the :class:`CacheStore`
Protocol so the cache can sit on top of either a plain filesystem
directory (``FileCacheStore``) or a workspace's private subsystem
storage (``WorkspaceCacheStore``).

The rectification spec (2026-05-09) introduced this split so workflow
no longer hardcodes a user-home cache directory; callers backed by a
workspace get content-addressed entries under
``<workspace_root>/.subsystems/workflow.cache/`` with workspace's
atomic-write guarantee.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from molexp.workspace import atomic_write_json

if TYPE_CHECKING:
    from molexp.workspace import Workspace

# Subsystem identity used by ``WorkspaceCacheStore``. The string is a
# *workflow-layer* convention; workspace only validates the shape.
WORKFLOW_CACHE_SUBSYSTEM_KIND = "workflow.cache"


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
    aware callers should reach for :class:`WorkspaceCacheStore`
    instead.
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


# ── Workspace-backed implementation ─────────────────────────────────────


class WorkspaceCacheStore:
    """Workspace-backed :class:`CacheStore`.

    Stores cache entries under
    ``<workspace_root>/.subsystems/workflow.cache/<key>.json`` via
    workspace's :class:`SubsystemStore`. Writes go through workspace's
    public :func:`atomic_write_json` helper so atomicity is workspace's
    guarantee, not the cache's reinvented one.

    The same workspace can back multiple :class:`Caching` instances —
    keys are content-addressed (snapshot key + input hash), so
    collisions across pipelines are vanishingly unlikely.
    """

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self._store = workspace.subsystem_store(WORKFLOW_CACHE_SUBSYSTEM_KIND)

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def store_dir(self) -> Path:
        # Lazily creates the directory on first call — matches the
        # rest of workspace's "I/O on first touch" rule.
        return self._store.dir()

    def read(self, key: str) -> str | None:
        path = _entry_path(self.store_dir, key)
        if not path.exists():
            return None
        try:
            return path.read_text()
        except OSError:
            return None

    def write(self, key: str, content: str) -> None:
        path = _entry_path(self.store_dir, key)
        # ``atomic_write_json`` accepts the structural top-type
        # ``object`` and serializes via ``json.dumps``. The cache
        # already has a JSON string; round-trip through ``json.loads``
        # so the on-disk format matches the file-store implementation
        # (``.json`` files containing canonical JSON).
        import json as _json

        atomic_write_json(path, _json.loads(content))

    def remove(self, key: str) -> bool:
        path = _entry_path(self.store_dir, key)
        if not path.exists():
            return False
        path.unlink(missing_ok=True)
        return True

    def keys(self) -> Iterable[str]:
        d = self.store_dir
        return tuple(p.stem for p in d.glob("*.json"))

    def access_time(self, key: str) -> float:
        path = _entry_path(self.store_dir, key)
        if not path.exists():
            return 0.0
        return path.stat().st_mtime

    def touch(self, key: str) -> None:
        path = _entry_path(self.store_dir, key)
        if path.exists():
            path.touch()

    def total_bytes(self) -> int:
        d = self.store_dir
        return sum(p.stat().st_size for p in d.glob("*.json"))

    def clear(self) -> int:
        d = self.store_dir
        count = 0
        for p in d.glob("*.json"):
            p.unlink(missing_ok=True)
            count += 1
        return count


__all__ = [
    "WORKFLOW_CACHE_SUBSYSTEM_KIND",
    "CacheStore",
    "FileCacheStore",
    "WorkspaceCacheStore",
]
