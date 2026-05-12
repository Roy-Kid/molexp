"""``CacheFolder`` — workspace-rooted content-addressed cache.

Stores entries as ``<workspace_root>/cache/<key>.json``. Schema-agnostic:
the folder vendors string blobs; the workflow layer's :class:`Caching`
owns the encoding.

The CacheStore adapter is exposed via :meth:`CacheFolder.as_cache_store`.
The adapter is a plain Python class — workflow's ``CacheStore`` is a
``runtime_checkable`` Protocol, so structural typing means workspace
needs no import edge to the workflow package. Callers obtain the
Protocol type from ``molexp.workflow.cache_store`` themselves.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from ..folder import Folder

WORKSPACE_CACHE_KIND = "workspace.cache"


def _entry_path(store_dir: Path, key: str) -> Path:
    return store_dir / f"{key}.json"


class CacheFolder(Folder):
    """Single-instance folder rooted at ``<workspace_root>/cache/``.

    Read/write methods are schema-agnostic (raw string content); callers
    are responsible for the on-disk JSON format. Entry paths are
    ``<root>/cache/<key>.json``; ``key`` is treated opaquely (the
    workflow cache passes a content-addressed sha256 hex digest).
    """

    def entry_path(self, key: str) -> Path:
        return self.path() / f"{key}.json"

    def read_entry(self, key: str) -> str | None:
        path = self.entry_path(key)
        if not path.exists():
            return None
        try:
            return path.read_text()
        except OSError:
            return None

    def write_entry(self, key: str, content: str) -> None:
        """Atomically replace ``<root>/cache/<key>.json`` with *content*.

        The bytes on disk are a canonical JSON serialization of
        ``json.loads(content)`` so that the workflow cache's
        machine-readable assertions (entry is valid JSON, ``snapshot_key``
        column present) keep working through the workspace path.
        """
        path = self.entry_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Round-trip the caller-supplied content through json.loads so
        # the persisted file is canonical JSON — matches the historic
        # ``WorkspaceCacheStore.write`` contract.
        payload = json.loads(content)
        fd, tmp_str = tempfile.mkstemp(
            dir=path.parent, suffix=".tmp", prefix=f".{path.stem}_"
        )
        tmp_path = Path(tmp_str)
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, indent=2, default=str)
            tmp_path.replace(path)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            raise

    def remove_entry(self, key: str) -> bool:
        path = self.entry_path(key)
        if not path.exists():
            return False
        path.unlink(missing_ok=True)
        return True

    def keys(self) -> Iterable[str]:
        dir_path = self._compute_path()
        if not dir_path.exists():
            return ()
        return tuple(p.stem for p in dir_path.glob("*.json"))

    def access_time(self, key: str) -> float:
        path = self.entry_path(key)
        if not path.exists():
            return 0.0
        return path.stat().st_mtime

    def touch(self, key: str) -> None:
        path = self.entry_path(key)
        if path.exists():
            path.touch()

    def total_bytes(self) -> int:
        dir_path = self._compute_path()
        if not dir_path.exists():
            return 0
        return sum(p.stat().st_size for p in dir_path.glob("*.json"))

    def clear(self) -> int:
        dir_path = self._compute_path()
        if not dir_path.exists():
            return 0
        count = 0
        for p in dir_path.glob("*.json"):
            p.unlink(missing_ok=True)
            count += 1
        return count

    def as_cache_store(self) -> _CacheFolderAdapter:
        """Return an adapter satisfying ``molexp.workflow.cache_store.CacheStore``.

        The adapter is a plain class — workflow's ``CacheStore`` is a
        ``runtime_checkable`` Protocol, so the return value structurally
        satisfies the interface without workspace importing workflow.
        Callers can ``isinstance(_, CacheStore)`` once they have the
        Protocol in hand.
        """
        return _CacheFolderAdapter(self)


class _CacheFolderAdapter:
    """Bridge a :class:`CacheFolder` to the workflow ``CacheStore`` Protocol.

    Structural typing (``runtime_checkable Protocol``) means no explicit
    base-class link is needed — matching method signatures is sufficient
    and keeps the workflow import edge from forming at module load.
    """

    def __init__(self, folder: CacheFolder) -> None:
        self._folder = folder

    @property
    def store_dir(self) -> Path:
        return self._folder.path()

    def read(self, key: str) -> str | None:
        return self._folder.read_entry(key)

    def write(self, key: str, content: str) -> None:
        self._folder.write_entry(key, content)

    def remove(self, key: str) -> bool:
        return self._folder.remove_entry(key)

    def keys(self) -> Iterable[str]:
        return self._folder.keys()

    def access_time(self, key: str) -> float:
        return self._folder.access_time(key)

    def touch(self, key: str) -> None:
        self._folder.touch(key)

    def total_bytes(self) -> int:
        return self._folder.total_bytes()

    def clear(self) -> int:
        return self._folder.clear()


__all__ = [
    "WORKSPACE_CACHE_KIND",
    "CacheFolder",
]
