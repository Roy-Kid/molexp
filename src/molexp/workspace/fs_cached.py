"""Caching :class:`FileSystem` decorator — lazy-download mirror for
remote workspaces.

Wraps an inner :class:`~molexp.workspace.fs.FileSystem` (only meaningful
for :class:`~molexp.workspace.fs_remote.RemoteFileSystem`) and maintains
a server-side mirror under ``<mirror_root>/files/...``.  Reads check the
mirror first; on miss they fetch from the inner FS, write the bytes
atomically to the mirror, and serve subsequent reads with zero round
trips for the configured TTL.  Mutations (write/rename/remove)
invalidate the affected entry before delegating.

Index files are not special-cased — they are just paths.  The eager
prefetch helper :func:`prefetch_workspace_indices` walks the workspace by
``listdir`` plus the per-entity ``workspace.json`` / ``project.json`` /
``experiment.json`` / ``run.json`` metadata files through
:meth:`read_text`, so the navigation tree is populated as a side-effect
of caching.  The entity ``*.json`` is the sole truth source; there is no
separate plural container-index chain.

Layer rule: lives in the workspace layer next to ``fs_local.py`` and
``fs_remote.py``; reaches only into sibling FS modules and the
:func:`atomic_write_json` primitive.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from .fs import FileSystem, PathArg, StatResult
from .fs_local import LocalFileSystem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .workspace import Workspace

__all__ = [
    "INDEX_FILE_NAMES",
    "CachedRemoteFileSystem",
    "PrefetchWarning",
    "prefetch_workspace_indices",
]

logger = logging.getLogger(__name__)

INDEX_FILE_NAMES: frozenset[str] = frozenset(
    {
        "workspace.json",
        "project.json",
        "experiment.json",
        "run.json",
    }
)
"""Files whose basename identifies them as a navigation-index artefact.

In molexp's workspace layout these singular names are an entity's own
metadata (``<child>/run.json`` etc.); the entity ``*.json`` is the sole
truth source for the navigation tree.  Their basenames double as the
``scope="indices"`` invalidation set, so a refresh drops cached
navigation metadata while sparing log/asset bytes.
"""

_SIDECAR_FILENAME = "_index.json"
_SIDECAR_VERSION = 1


@dataclass(frozen=True)
class _Entry:
    """One cached file/dir/missing record."""

    size: int
    mtime: float
    fetched_at: float
    kind: str  # "file" | "dir" | "missing"


@dataclass(frozen=True)
class _DirEntry:
    """One cached listdir result."""

    names: tuple[str, ...]
    fetched_at: float


@dataclass(frozen=True)
class PrefetchWarning:
    """One node that failed during :func:`prefetch_workspace_indices`."""

    path: str
    reason: str


class CachedRemoteFileSystem:
    """Lazy-download mirror over any :class:`FileSystem`.

    Reads first consult an in-memory ``_index`` whose entries are valid
    for ``ttl_seconds``; a fresh entry returns mirror bytes without a
    remote round trip.  Mutations always go to the inner FS and
    invalidate the cached entry.

    The mirror layout reflects the remote path verbatim (leading ``/``
    stripped) under ``<mirror_root>/files/``, so a remote path
    ``/home/me/run/log.txt`` ends up at ``<mirror_root>/files/home/me/
    run/log.txt``.  This stays debuggable and lets ``find`` walk the
    mirror.

    Args:
        inner: The :class:`FileSystem` to cache.
        mirror_root: Local directory holding the mirror.  Created on
            first write.
        ttl_seconds: How long a cached entry is considered fresh.  ``0``
            disables the fast path — every read re-stats the inner FS,
            but already-fetched mirror bytes are still returned when
            ``stat.mtime`` matches the cached entry.
    """

    def __init__(
        self,
        inner: FileSystem,
        *,
        mirror_root: Path | str,
        ttl_seconds: int = 300,
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be >= 0")
        self._inner = inner
        self._local = LocalFileSystem()
        self._mirror_root = Path(mirror_root)
        self._files_root = self._mirror_root / "files"
        self._ttl_seconds = ttl_seconds
        self._index: dict[str, _Entry] = {}
        self._dir_index: dict[str, _DirEntry] = {}
        self._sidecar = self._mirror_root / _SIDECAR_FILENAME
        self._load_sidecar()

    # ── Test-only introspection ─────────────────────────────────────────

    @property
    def inner(self) -> FileSystem:
        return self._inner

    @property
    def mirror_root(self) -> Path:
        return self._mirror_root

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    def cached_paths(self) -> list[str]:
        """Snapshot of cached file/dir/missing paths — handy in tests."""
        return list(self._index.keys())

    # ── Path operations (always delegate; no I/O) ───────────────────────

    def join(self, *parts: PathArg) -> str:
        return self._inner.join(*parts)

    def dirname(self, path: PathArg) -> str:
        return self._inner.dirname(path)

    def basename(self, path: PathArg) -> str:
        return self._inner.basename(path)

    def resolve(self, path: PathArg) -> str:
        return self._inner.resolve(path)

    def is_absolute(self, path: PathArg) -> bool:
        return self._inner.is_absolute(path)

    # ── Existence / type ────────────────────────────────────────────────

    def exists(self, path: PathArg) -> bool:
        key = self.resolve(path)
        entry = self._fresh_entry(key)
        if entry is not None:
            return entry.kind != "missing"
        result = self._inner.exists(key)
        if not result:
            # Negative cache: future ``exists`` returns False without SSH.
            self._record(key, kind="missing", size=0, mtime=0.0)
        return result

    def is_dir(self, path: PathArg) -> bool:
        key = self.resolve(path)
        entry = self._fresh_entry(key)
        if entry is not None:
            return entry.kind == "dir"
        result = self._inner.is_dir(key)
        if result:
            self._record(key, kind="dir", size=0, mtime=time.time())
        return result

    def is_file(self, path: PathArg) -> bool:
        key = self.resolve(path)
        entry = self._fresh_entry(key)
        if entry is not None:
            return entry.kind == "file"
        result = self._inner.is_file(key)
        if result:
            # Don't fetch yet — just record what we learned.
            stat_value = self._safe_stat(key)
            if stat_value is not None:
                self._record(
                    key,
                    kind="file",
                    size=stat_value.size,
                    mtime=stat_value.mtime,
                )
        return result

    # ── Directory operations ────────────────────────────────────────────

    def mkdir(self, path: PathArg, *, parents: bool = True, exist_ok: bool = True) -> None:
        key = self.resolve(path)
        self._inner.mkdir(key, parents=parents, exist_ok=exist_ok)
        self._record(key, kind="dir", size=0, mtime=time.time())
        self._invalidate_dir(self._inner.dirname(key))

    def listdir(self, path: PathArg) -> list[str]:
        key = self.resolve(path)
        cached = self._fresh_dir(key)
        if cached is not None:
            return list(cached.names)
        names = self._inner.listdir(key)
        self._dir_index[key] = _DirEntry(names=tuple(names), fetched_at=time.time())
        self._persist_sidecar()
        return names

    def glob(self, path: PathArg, pattern: str) -> Iterable[str]:
        # Glob is intentionally uncached — patterns are open-ended and
        # caching them risks staleness on every directory change.
        return self._inner.glob(path, pattern)

    def rglob(self, path: PathArg, pattern: str) -> Iterable[str]:
        return self._inner.rglob(path, pattern)

    # ── Read ────────────────────────────────────────────────────────────

    def read_text(self, path: PathArg, encoding: str = "utf-8") -> str:
        return self.read_bytes(path).decode(encoding)

    def read_bytes(self, path: PathArg) -> bytes:
        key = self.resolve(path)
        mirror_path = self._mirror_for(key)
        entry = self._fresh_entry(key)
        if entry is not None and entry.kind == "file" and self._local.exists(mirror_path):
            return self._local.read_bytes(mirror_path)
        if entry is not None and entry.kind == "missing":
            raise FileNotFoundError(key)
        # Miss or stale — fetch fresh.
        try:
            data = self._inner.read_bytes(key)
        except FileNotFoundError:
            self._record(key, kind="missing", size=0, mtime=0.0)
            raise
        stat_value = self._safe_stat(key)
        size = len(data) if stat_value is None else stat_value.size
        mtime = time.time() if stat_value is None else stat_value.mtime
        self._write_mirror(mirror_path, data)
        self._record(key, kind="file", size=size, mtime=mtime)
        return data

    def open(self, path: PathArg, mode: str = "r", encoding: str = "utf-8") -> IO[Any]:  # noqa: ARG002 — `mode` kept to mirror RemoteFileSystem.open's signature
        # Mirror RemoteFileSystem's behaviour: read-only string buffer.
        import io

        return io.StringIO(self.read_text(path, encoding=encoding))

    # ── Write ───────────────────────────────────────────────────────────

    def write_text(self, path: PathArg, content: str, *, mode: int = 0o600) -> None:
        key = self.resolve(path)
        self._invalidate(key)
        self._inner.write_text(key, content, mode=mode)

    def write_bytes(self, path: PathArg, content: bytes, *, mode: int = 0o600) -> None:
        key = self.resolve(path)
        self._invalidate(key)
        self._inner.write_bytes(key, content, mode=mode)

    # ── Mutations ───────────────────────────────────────────────────────

    def rename(self, src: PathArg, dst: PathArg) -> None:
        src_key = self.resolve(src)
        dst_key = self.resolve(dst)
        self._invalidate(src_key)
        self._invalidate(dst_key)
        self._inner.rename(src_key, dst_key)

    def remove(self, path: PathArg, *, recursive: bool = False) -> None:
        key = self.resolve(path)
        self._invalidate(key, recursive=recursive)
        self._inner.remove(key, recursive=recursive)

    def copy(self, src: PathArg, dst: PathArg) -> None:
        dst_key = self.resolve(dst)
        self._invalidate(dst_key)
        self._inner.copy(src, dst_key)

    def copytree(self, src: PathArg, dst: PathArg, *, dirs_exist_ok: bool = False) -> None:
        dst_key = self.resolve(dst)
        self._invalidate(dst_key, recursive=True)
        self._inner.copytree(src, dst_key, dirs_exist_ok=dirs_exist_ok)

    # ── Metadata ────────────────────────────────────────────────────────

    def stat(self, path: PathArg) -> StatResult:
        key = self.resolve(path)
        entry = self._fresh_entry(key)
        if entry is not None and entry.kind != "missing":
            return StatResult(
                size=entry.size,
                mtime=entry.mtime,
                is_dir=entry.kind == "dir",
                is_file=entry.kind == "file",
            )
        result = self._inner.stat(key)
        kind = "dir" if result.is_dir else "file" if result.is_file else "missing"
        self._record(key, kind=kind, size=result.size, mtime=result.mtime)
        return result

    def lstat(self, path: PathArg) -> StatResult:
        return self.stat(path)

    def touch(self, path: PathArg) -> None:
        key = self.resolve(path)
        self._invalidate(key)
        self._inner.touch(key)

    def chmod(self, path: PathArg, mode: int) -> None:
        self._inner.chmod(path, mode)

    def getsize(self, path: PathArg) -> int:
        return self.stat(path).size

    # ── Symlinks ────────────────────────────────────────────────────────

    def symlink(self, src: PathArg, dst: PathArg) -> None:
        dst_key = self.resolve(dst)
        self._invalidate(dst_key)
        self._inner.symlink(src, dst_key)

    # ── Atomic I/O ──────────────────────────────────────────────────────

    def atomic_write_json(self, path: PathArg, data: object) -> None:
        key = self.resolve(path)
        self._invalidate(key)
        self._inner.atomic_write_json(key, data)

    def atomic_write_text(self, path: PathArg, content: str, *, encoding: str = "utf-8") -> None:
        key = self.resolve(path)
        self._invalidate(key)
        self._inner.atomic_write_text(key, content, encoding=encoding)

    # ── Cache control ───────────────────────────────────────────────────

    def invalidate(
        self,
        path: PathArg | None = None,
        *,
        scope: str = "all",
    ) -> int:
        """Drop cached entries; return the number dropped.

        Args:
            path: Drop only this entry (and its descendants if a dir).
                ``None`` drops based on ``scope``.
            scope: ``"all"`` drops every entry and removes the mirror
                directory.  ``"indices"`` drops only entries whose
                basename is in :data:`INDEX_FILE_NAMES` (lets the UI
                refresh navigation without throwing away cached log
                bytes).
        """
        if path is not None:
            key = self.resolve(path)
            return self._invalidate(key, recursive=True)
        if scope == "indices":
            keys = [k for k in self._index if self._inner.basename(k) in INDEX_FILE_NAMES]
            for key in keys:
                self._invalidate(key)
            return len(keys)
        if scope == "all":
            count = len(self._index)
            self._index.clear()
            self._dir_index.clear()
            if self._files_root.exists():
                with contextlib.suppress(OSError):
                    shutil.rmtree(self._files_root)
            self._persist_sidecar()
            return count
        raise ValueError(f"unknown scope {scope!r}")

    # ── Internals ───────────────────────────────────────────────────────

    def _fresh_entry(self, key: str) -> _Entry | None:
        entry = self._index.get(key)
        if entry is None:
            return None
        if self._ttl_seconds == 0:
            return None
        if time.time() - entry.fetched_at > self._ttl_seconds:
            return None
        return entry

    def _fresh_dir(self, key: str) -> _DirEntry | None:
        entry = self._dir_index.get(key)
        if entry is None:
            return None
        if self._ttl_seconds == 0:
            return None
        if time.time() - entry.fetched_at > self._ttl_seconds:
            return None
        return entry

    def _record(self, key: str, *, kind: str, size: int, mtime: float) -> None:
        self._index[key] = _Entry(
            size=size,
            mtime=mtime,
            fetched_at=time.time(),
            kind=kind,
        )
        self._persist_sidecar()

    def _invalidate(self, key: str, *, recursive: bool = False) -> int:
        dropped = 0
        if key in self._index:
            del self._index[key]
            dropped += 1
        if recursive:
            prefix = key.rstrip("/") + "/"
            for k in list(self._index):
                if k.startswith(prefix):
                    del self._index[k]
                    dropped += 1
            for k in list(self._dir_index):
                if k == key or k.startswith(prefix):
                    del self._dir_index[k]
        # Always invalidate the parent dir listing.
        self._invalidate_dir(self._inner.dirname(key))
        # Best-effort mirror eviction.
        mirror_path = self._mirror_for(key)
        if self._local.exists(mirror_path):
            with contextlib.suppress(OSError):
                if recursive and self._local.is_dir(mirror_path):
                    shutil.rmtree(mirror_path)
                else:
                    self._local.remove(mirror_path)
        self._persist_sidecar()
        return dropped

    def _invalidate_dir(self, dir_key: str) -> None:
        self._dir_index.pop(dir_key, None)

    def _mirror_for(self, abs_path: str) -> Path:
        # Strip any leading slashes so we stay inside files/.
        relative = os.fspath(abs_path).lstrip("/")
        return self._files_root / relative

    def _write_mirror(self, mirror_path: Path, data: bytes) -> None:
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = mirror_path.with_suffix(mirror_path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, mirror_path)  # noqa: PTH105

    def _safe_stat(self, key: str) -> StatResult | None:
        try:
            return self._inner.stat(key)
        except Exception:
            return None

    def _load_sidecar(self) -> None:
        if not self._sidecar.exists():
            return
        try:
            raw = json.loads(self._sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "cache sidecar at %s unreadable; starting empty (%s)", self._sidecar, exc
            )
            return
        if not isinstance(raw, dict) or raw.get("version") != _SIDECAR_VERSION:
            logger.warning("cache sidecar at %s has wrong version; starting empty", self._sidecar)
            return
        entries = raw.get("entries", {}) or {}
        for key, payload in entries.items():
            try:
                self._index[key] = _Entry(**payload)
            except TypeError:
                continue
        dirs = raw.get("dirs", {}) or {}
        for key, payload in dirs.items():
            try:
                names = tuple(payload.get("names", ()))
                fetched_at = float(payload.get("fetched_at", 0.0))
                self._dir_index[key] = _DirEntry(names=names, fetched_at=fetched_at)
            except (AttributeError, TypeError, ValueError):
                continue

    def _persist_sidecar(self) -> None:
        payload = {
            "version": _SIDECAR_VERSION,
            "ttl_seconds": self._ttl_seconds,
            "entries": {k: asdict(v) for k, v in self._index.items()},
            "dirs": {
                k: {"names": list(v.names), "fetched_at": v.fetched_at}
                for k, v in self._dir_index.items()
            },
        }
        self._mirror_root.mkdir(parents=True, exist_ok=True)
        tmp = self._sidecar.with_suffix(self._sidecar.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self._sidecar)  # noqa: PTH105
        except OSError as exc:
            logger.warning("cache sidecar write failed at %s: %s", self._sidecar, exc)
            with contextlib.suppress(OSError):
                tmp.unlink()


@dataclass
class _PrefetchState:
    warnings: list[PrefetchWarning] = field(default_factory=list)


def prefetch_workspace_indices(workspace: Workspace) -> list[PrefetchWarning]:
    """Walk the workspace's entity metadata through ``workspace._fs``.

    Reads (in order):

    1. ``<root>/workspace.json`` — workspace metadata.
    2. For each project under ``<root>/projects/``: that project's own
       ``project.json`` metadata.
    3. For each experiment under ``<project>/experiments/``: its
       ``experiment.json`` metadata.
    4. For each run under ``<experiment>/runs/``: its ``run.json``
       metadata.

    Child names come from a ``listdir`` of each container directory; the
    sibling children-index file (``project.json`` / ``experiment.json`` /
    ``run.json`` at the *parent* path) is read once to warm the cache but
    is no longer the source of truth — the entity ``*.json`` is.  Every
    read flows through ``workspace._fs.read_text``; if the FS is a
    :class:`CachedRemoteFileSystem`, the entries are cached as a side
    effect, so subsequent navigation clicks hit zero SSH.

    Missing or unreadable nodes degrade to :class:`PrefetchWarning`
    entries; the walk continues so a single bad project does not blank
    the whole tree.

    Returns:
        A flat list of warnings in walk order.  Empty list on a clean
        walk.
    """
    state = _PrefetchState()
    fs = workspace._fs
    root = str(workspace.root)
    _safe_read(fs, fs.join(root, "workspace.json"), state)
    projects_dir = fs.join(root, "projects")
    project_names = _read_container_children(
        fs,
        container_dir=projects_dir,
        index_path=fs.join(root, "project.json"),
        per_child_metadata="project.json",
        state=state,
    )
    for project_name in project_names:
        project_dir = fs.join(projects_dir, project_name)
        experiments_dir = fs.join(project_dir, "experiments")
        experiment_names = _read_container_children(
            fs,
            container_dir=experiments_dir,
            index_path=fs.join(project_dir, "experiment.json"),
            per_child_metadata="experiment.json",
            state=state,
        )
        for experiment_name in experiment_names:
            experiment_dir = fs.join(experiments_dir, experiment_name)
            runs_dir = fs.join(experiment_dir, "runs")
            _read_container_children(
                fs,
                container_dir=runs_dir,
                index_path=fs.join(experiment_dir, "run.json"),
                per_child_metadata="run.json",
                state=state,
            )
    return state.warnings


def _safe_read(
    fs: FileSystem,
    path: str,
    state: _PrefetchState,
    *,
    warn_on_missing: bool = True,
) -> str | None:
    try:
        return fs.read_text(path)
    except FileNotFoundError as exc:
        if warn_on_missing:
            state.warnings.append(PrefetchWarning(path=path, reason=f"not found: {exc}"))
        return None
    except Exception as exc:
        state.warnings.append(PrefetchWarning(path=path, reason=str(exc)))
        return None


def _read_container_children(
    fs: FileSystem,
    *,
    container_dir: str,
    index_path: str,
    per_child_metadata: str,
    state: _PrefetchState,
) -> list[str]:
    """Warm the children-index, then list the container directly.

    The sibling children-index file at *index_path* is read once to warm
    the cache (navigation reads it back), but the authoritative child
    names come from a ``listdir`` of *container_dir* plus a per-child
    metadata probe — the entity ``*.json`` is the sole truth source, the
    run subdir name (``run-<id>``) differs from any index key, and the
    index is rebuilt lazily so a fresh hierarchy may lack it.

    A *missing* index is silent (the directory scan covers it); any
    non-``FileNotFoundError`` transport error on the index read still
    surfaces as a warning.  Returns the names of subdirectories whose
    metadata read succeeded; per-child failures are recorded as warnings
    and omitted.
    """
    _safe_read(fs, index_path, state, warn_on_missing=False)
    try:
        names = fs.listdir(container_dir)
    except FileNotFoundError:
        return []
    except Exception as exc:
        state.warnings.append(PrefetchWarning(path=container_dir, reason=str(exc)))
        return []
    healthy: list[str] = []
    for name in names:
        meta_path = fs.join(container_dir, name, per_child_metadata)
        if _safe_read(fs, meta_path, state) is not None:
            healthy.append(name)
    return healthy
