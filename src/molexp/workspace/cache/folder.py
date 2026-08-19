"""``CacheFolder`` — workspace-rooted content-addressed cache."""

from __future__ import annotations

import json
from collections.abc import Iterable

from molexp.path import Path

from ..folder import Folder

WORKSPACE_CACHE_KIND = "workspace.cache"


class CacheFolder(Folder):
    """Single-instance folder rooted at ``<workspace_root>/cache/``."""

    def entry_path(self, key: str) -> Path:
        return Path(self._disk().join(self.path(), f"{key}.json"))

    def read_entry(self, key: str) -> str | None:
        p = self.entry_path(key)
        if not self._disk().exists(p):
            return None
        try:
            return self._disk().read_text(p)
        except OSError:
            return None

    def write_entry(self, key: str, content: str) -> None:
        p = self.entry_path(key)
        payload = json.loads(content)
        self._disk().atomic_write_json(p, payload)

    def remove_entry(self, key: str) -> bool:
        p = self.entry_path(key)
        if not self._disk().exists(p):
            return False
        self._disk().remove(p)
        return True

    def keys(self) -> Iterable[str]:
        dir_path = self.resolve()
        if not self._disk().is_dir(dir_path):
            return ()
        return tuple(
            self._disk().basename(p).removesuffix(".json")
            for p in self._disk().glob(dir_path, "*.json")
        )

    def access_time(self, key: str) -> float:
        p = self.entry_path(key)
        if not self._disk().exists(p):
            return 0.0
        return self._disk().stat(p).mtime

    def touch(self, key: str) -> None:
        p = self.entry_path(key)
        if self._disk().exists(p):
            self._disk().touch(p)

    def total_bytes(self) -> int:
        dir_path = self.resolve()
        if not self._disk().is_dir(dir_path):
            return 0
        return sum(self._disk().stat(p).size for p in self._disk().glob(dir_path, "*.json"))

    def clear(self) -> int:
        dir_path = self.resolve()
        if not self._disk().is_dir(dir_path):
            return 0
        count = 0
        for p in self._disk().glob(dir_path, "*.json"):
            self._disk().remove(p)
            count += 1
        return count

    def blob_path(self, content_hash: str) -> Path:
        safe = content_hash.replace(":", "-")
        return Path(self._disk().join(self.path(), "blobs", safe))

    def put_blob(self, content_hash: str, data: bytes) -> None:
        p = self.blob_path(content_hash)
        parent = Path(self._disk().dirname(p)) if hasattr(self._disk(), "dirname") else p.parent
        self._disk().mkdir(str(parent), parents=True, exist_ok=True)
        if self._disk().exists(p):
            return
        self._disk().write_bytes(p, data)

    def get_blob(self, content_hash: str) -> bytes | None:
        p = self.blob_path(content_hash)
        if not self._disk().exists(p):
            return None
        try:
            return self._disk().read_bytes(p)
        except OSError:
            return None

    def as_cache_store(self) -> _CacheFolderAdapter:
        return _CacheFolderAdapter(self)


class _CacheFolderAdapter:
    """Bridge a :class:`CacheFolder` to the workflow ``CacheStore`` Protocol."""

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

    def put_blob(self, content_hash: str, data: bytes) -> None:
        self._folder.put_blob(content_hash, data)

    def get_blob(self, content_hash: str) -> bytes | None:
        return self._folder.get_blob(content_hash)


__all__ = [
    "WORKSPACE_CACHE_KIND",
    "CacheFolder",
]
