"""Caching :class:`FileSystem` decorator — lazy-download mirror for
remote workspaces.

Wraps an inner :class:`~molexp.workspace.fs.FileSystem` (only meaningful
for :class:`~molexp.workspace.fs_remote.RemoteFileSystem`) and maintains
a server-side mirror under ``<mirror_root>/files/...``.

**Pin-until-refresh policy** (default): once a path is in the local
index/mirror it is trusted forever.  Age / ``ttl_seconds`` does **not**
trigger automatic revalidation — the operator refreshes via
``POST /api/workspace/cache/refresh`` (or ``invalidate``).  Cache misses
still go to the remote FS and populate the mirror.

``ttl_seconds=0`` is an opt-in strict mode: every read re-stats the remote
and reuses mirror bytes only when mtime/size still match.

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
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
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
    "IndexProgress",
    "PrefetchWarning",
    "prefetch_workspace_indices",
]


# Outside-in parallel prefetch: concurrent SSH ops per level. Override with
# ``MOLEXP_PREFETCH_WORKERS`` (1 = serial, useful in tests).
_DEFAULT_PREFETCH_WORKERS = 8

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


@dataclass
class IndexProgress:
    """Live remote-index progress for the status bar.

    *counting* — recursive total is still being computed.
    *fetching* — ``done/total`` advance as files are force-fetched.
    *done* / *error* — terminal.
    """

    phase: str = "idle"  # idle | counting | fetching | done | error
    total: int = 0
    done: int = 0
    message: str = ""

    @property
    def percent(self) -> float | None:
        if self.total <= 0:
            return None
        return min(100.0, 100.0 * self.done / self.total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "total": self.total,
            "done": self.done,
            "percent": self.percent,
            "message": self.message,
        }


class CachedRemoteFileSystem:
    """Lazy-download mirror over any :class:`FileSystem`.

    **Default (``ttl_seconds > 0``)**: pin-until-refresh.  Any path present
    in the sidecar/mirror is served locally with **zero** remote I/O until
    the operator invalidates/refreshes.  Mutations still go to the inner
    FS and invalidate the affected entry.

    **Strict (``ttl_seconds == 0``)**: every read re-stats the remote and
    reuses mirror bytes only when mtime/size match (no silent pin).

    The mirror layout reflects the remote path verbatim (leading ``/``
    stripped) under ``<mirror_root>/files/``, so a remote path
    ``/home/me/run/log.txt`` ends up at ``<mirror_root>/files/home/me/
    run/log.txt``.  This stays debuggable and lets ``find`` walk the
    mirror.

    Args:
        inner: The :class:`FileSystem` to cache.
        mirror_root: Local directory holding the mirror.  Created on
            first write.
        ttl_seconds: ``>0`` (default) = pin-until-refresh.  ``0`` =
            revalidate via remote ``stat`` on every read.
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
        # Sidecar write batching: while ``_defer_persist`` is set (inside
        # ``batched()``), per-op writes only mark ``_sidecar_dirty`` and the
        # full serialization happens once on batch exit — turning a bulk walk
        # (e.g. ``prefetch_workspace_indices``) from O(records²) into O(records).
        self._defer_persist = False
        self._sidecar_dirty = False
        # Lifecycle flags — a brand-new remote root has no sidecar; that is
        # normal. ``connect`` / ``index`` (or ``prepare``) flip these once
        # the local mirror is ready and navigation metadata is warm.
        self._connected = False
        self._indexed = False
        self._remote_root: str | None = None
        self._lock = threading.RLock()
        self._index_thread: threading.Thread | None = None
        self._indexing = False
        self._progress = IndexProgress()
        # Per-thread: active refresh bypasses pin and re-fetches from remote.
        # UI threads keep serving the pinned mirror while a refresh runs.
        self._tls = threading.local()
        # Always own the local mirror tree up-front so a missing
        # ``_index.json`` never surfaces as "Path not found" on first write.
        self._ensure_mirror_dirs()
        self._load_sidecar()
        if self._index or self._dir_index:
            # Survived a previous session — treat as already indexed until
            # the caller re-runs ``index()`` / invalidates.
            self._indexed = True

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

    @property
    def connected(self) -> bool:
        """True after a successful :meth:`connect` (remote root reachable)."""
        return self._connected

    @property
    def indexed(self) -> bool:
        """True after :meth:`index` / :meth:`connect_and_index` (or a loaded sidecar)."""
        return self._indexed

    @property
    def ready(self) -> bool:
        """True when navigation can be served from the local mirror/index.

        SSH may still be deferred (warm reopen) — :attr:`connected` is the
        probe flag; :attr:`ready` is "UI can load the tree".
        """
        return self._indexed

    @property
    def indexing(self) -> bool:
        """True while a background :meth:`schedule_index` walk is in flight."""
        return self._indexing

    @property
    def progress(self) -> IndexProgress:
        """Snapshot of the live index walk (safe to read from any thread)."""
        with self._lock:
            p = self._progress
            return IndexProgress(
                phase=p.phase,
                total=p.total,
                done=p.done,
                message=p.message,
            )

    def _set_progress(
        self,
        *,
        phase: str | None = None,
        total: int | None = None,
        done: int | None = None,
        message: str | None = None,
        inc_done: int = 0,
    ) -> None:
        with self._lock:
            if phase is not None:
                self._progress.phase = phase
            if total is not None:
                self._progress.total = max(0, total)
            if done is not None:
                self._progress.done = max(0, done)
            if inc_done:
                self._progress.done = max(0, self._progress.done + inc_done)
            if message is not None:
                self._progress.message = message

    def cached_paths(self) -> list[str]:
        """Snapshot of cached file/dir/missing paths — handy in tests."""
        return list(self._index.keys())

    # ── Connect / index lifecycle ───────────────────────────────────────

    def _ensure_mirror_dirs(self) -> None:
        """Create ``mirror_root/`` and ``mirror_root/files/`` if missing."""
        self._mirror_root.mkdir(parents=True, exist_ok=True)
        self._files_root.mkdir(parents=True, exist_ok=True)

    def _ensure_connected(self) -> None:
        """Open SSH on first cache miss (warm reopen defers the probe).

        When neither :meth:`prepare` nor :meth:`connect` has recorded a root
        (unit tests / direct use), skip the probe and let the inner FS answer.
        """
        if self._connected:
            return
        root = self._remote_root
        if root is None:
            return
        self.connect(root)

    def connect(self, root: str) -> None:
        """Probe the remote root and materialise an empty local index if needed.

        A first-time workspace has no ``_index.json`` — that is expected.
        We create the local mirror dirs and write an empty sidecar so later
        cache records never fail with "No such file or directory" on the
        sidecar rename. Re-entrant / idempotent.
        """
        self._remote_root = root
        self._ensure_mirror_dirs()
        try:
            reachable = self._inner.exists(root) or self._inner.is_dir(root)
        except Exception as exc:
            self._connected = False
            raise ConnectionError(f"remote root unreachable: {root}: {exc}") from exc
        if not reachable:
            self._connected = False
            raise FileNotFoundError(f"remote root not found: {root}")
        # Missing sidecar is normal — write current in-memory state (often empty).
        if not self._sidecar.exists():
            self._write_sidecar()
        self._connected = True

    @contextlib.contextmanager
    def force_fetch(self) -> Iterator[None]:
        """Bypass pin for this thread — every read/listdir hits the remote.

        Used by active refreshes. Concurrent UI threads keep serving the
        pinned mirror (their ``_tls.force_fetch`` stays false).
        """
        prev = getattr(self._tls, "force_fetch", False)
        self._tls.force_fetch = True
        try:
            yield
        finally:
            self._tls.force_fetch = prev

    def count_remote_files(self, root: str) -> int:
        """Recursive file count under *root* (remote ``find -type f | wc -l``).

        Falls back to a BFS listdir walk when the transport has no ``run``.
        """
        inner = self._inner
        transport = getattr(inner, "_t", None)
        if transport is not None and hasattr(transport, "run"):
            # Single RTT: total file count as the progress denominator.
            import shlex

            try:
                cmd = f"find {shlex.quote(root)} -type f 2>/dev/null | wc -l"
                result = transport.run(["bash", "-lc", cmd])
                # molq Transport.run return shapes vary — accept str/bytes/obj.
                out = getattr(result, "stdout", None)
                if out is None:
                    out = result if isinstance(result, (str, bytes, int)) else ""
                if isinstance(out, bytes):
                    out = out.decode("utf-8", errors="replace")
                text = str(out).strip().splitlines()
                if text:
                    return max(0, int(text[-1].strip()))
            except Exception:
                logger.debug(
                    "remote find|wc failed for %s; falling back to BFS", root, exc_info=True
                )

        # BFS fallback (local FS / broken transport.run).
        total = 0
        stack = [root]
        while stack:
            cur = stack.pop()
            try:
                if not self._inner.is_dir(cur):
                    if self._inner.is_file(cur):
                        total += 1
                    continue
                for name in self._inner.listdir(cur):
                    if name.startswith("."):
                        continue
                    child = self._inner.join(cur, name)
                    try:
                        if self._inner.is_dir(child):
                            stack.append(child)
                        elif self._inner.is_file(child):
                            total += 1
                    except Exception:
                        continue
            except Exception:
                continue
        return total

    def index(self, workspace: Workspace) -> list[PrefetchWarning]:
        """Actively refresh navigation metadata from remote (blocking).

        Always force-fetches (does not trust pin). Outside-in parallel walk
        via :func:`prefetch_workspace_indices`. Sets :attr:`indexed`.

        Progress for the status bar:

        1. *counting* — recursive total file count under the root
        2. *fetching* — ``done/total`` as entity metadata is force-fetched
        3. *done*
        """
        self._remote_root = str(workspace.root)
        if not self._connected:
            self.connect(str(workspace.root))
        root = str(workspace.root)
        self._set_progress(phase="counting", total=0, done=0, message="Counting remote files…")
        try:
            total = self.count_remote_files(root)
        except Exception as exc:
            logger.warning("count_remote_files failed: %s", exc)
            total = 0
        self._set_progress(
            phase="fetching",
            total=max(total, 1),
            done=0,
            message=f"Syncing remote tree (0/{max(total, 1)})…",
        )
        try:
            with self.force_fetch():
                warnings = prefetch_workspace_indices(
                    workspace,
                    on_file=self._on_index_file,
                )
            # Prefetch uses batched(); flush guarantees the sidecar is on disk.
            self.flush()
            if not self._sidecar.exists():
                self._write_sidecar()
            self._indexed = True
            self._indexing = False
            # Snap to 100% so a total that over-counted still completes.
            with self._lock:
                tot = max(self._progress.total, self._progress.done, 1)
                self._progress.total = tot
                self._progress.done = tot
            self._set_progress(phase="done", message="Remote index ready")
            return warnings
        except Exception as exc:
            self._indexing = False
            self._set_progress(phase="error", message=f"Index failed: {exc}")
            raise

    def _on_index_file(self, _path: str) -> None:
        """Progress tick for each file force-fetched during index."""
        with self._lock:
            self._progress.done += 1
            done = self._progress.done
            total = max(self._progress.total, done)
            # If we discover more entity files than the find total predicted
            # (e.g. race / find missed), grow the denominator.
            if done > self._progress.total:
                self._progress.total = done
                total = done
            self._progress.message = f"Syncing remote tree ({done}/{total})…"

    def schedule_refresh(self, workspace: Workspace) -> None:
        """Run :meth:`index` on a daemon thread (non-blocking active refresh).

        Always starts a new force-fetch when idle — linking a remote must
        re-pull even if a previous pin exists. Idempotent while a walk is
        already in flight.
        """
        with self._lock:
            if self._indexing:
                return
            if self._index_thread is not None and self._index_thread.is_alive():
                return
            self._indexing = True
            root = str(workspace.root)
            self._remote_root = root

            def _run() -> None:
                try:
                    self.index(workspace)
                except Exception:
                    logger.exception(
                        "background remote index failed for %s — use cache/refresh",
                        root,
                    )
                    self._indexing = False
                    self._set_progress(phase="error", message="Remote index failed")

            self._index_thread = threading.Thread(
                target=_run,
                name="molexp-remote-index",
                daemon=True,
            )
            self._index_thread.start()

    # Back-compat alias
    schedule_index = schedule_refresh

    def prepare(
        self,
        workspace: Workspace,
        *,
        block_index: bool = False,
        refresh_on_open: bool = True,
    ) -> list[PrefetchWarning]:
        """Open path for ``molexp serve`` / API.

        * **On open / link** (``refresh_on_open=True``): always force-fetch
          from remote (even when a pin exists). Default is **async** so the
          UI can poll :attr:`progress` for a file-count progress bar;
          pass ``block_index=True`` to wait for the walk (CLI / tests).
        * **Cold**: probe SSH first, then the same refresh path.
        """
        self._remote_root = str(workspace.root)
        if not refresh_on_open:
            # Pure pin serve — no probe, no walk (tests / warm passive reopen).
            return []
        if not self._connected:
            self.connect(str(workspace.root))
        if block_index:
            return self.index(workspace)
        # Async force-refresh — UI polls GET /api/workspace/cache/status.
        self.schedule_refresh(workspace)
        return []

    def connect_and_index(self, workspace: Workspace) -> list[PrefetchWarning]:
        """Connect + build index synchronously (blocking). Prefer :meth:`prepare`."""
        return self.prepare(workspace, block_index=True, refresh_on_open=True)

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
        entry = self._pinned_entry(key)
        if entry is not None:
            return entry.kind != "missing"
        self._ensure_connected()
        result = self._inner.exists(key)
        if not result:
            # Negative cache: future ``exists`` returns False without SSH.
            self._record(key, kind="missing", size=0, mtime=0.0)
        return result

    def is_dir(self, path: PathArg) -> bool:
        key = self.resolve(path)
        entry = self._pinned_entry(key)
        if entry is not None:
            return entry.kind == "dir"
        self._ensure_connected()
        result = self._inner.is_dir(key)
        if result:
            self._record(key, kind="dir", size=0, mtime=time.time())
        return result

    def is_file(self, path: PathArg) -> bool:
        key = self.resolve(path)
        entry = self._pinned_entry(key)
        if entry is not None:
            return entry.kind == "file"
        self._ensure_connected()
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
        cached = self._pinned_dir(key)
        if cached is not None:
            return list(cached.names)
        self._ensure_connected()
        names = self._inner.listdir(key)
        with self._lock:
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
        entry = self._pinned_entry(key)
        if entry is not None and entry.kind == "file" and self._local.exists(mirror_path):
            return self._local.read_bytes(mirror_path)
        if entry is not None and entry.kind == "missing":
            raise FileNotFoundError(key)
        # Strict mode (ttl=0): revalidate via stat; serve mirror if unchanged.
        known = self._index.get(key)
        if (
            self._ttl_seconds == 0
            and known is not None
            and known.kind == "file"
            and self._local.exists(mirror_path)
        ):
            self._ensure_connected()
            if self._revalidate_file_entry(key, known):
                return self._local.read_bytes(mirror_path)
        # Miss (or strict revalidation failed) — fetch from remote.
        self._ensure_connected()
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
        entry = self._pinned_entry(key)
        if entry is not None and entry.kind != "missing":
            return StatResult(
                size=entry.size,
                mtime=entry.mtime,
                is_dir=entry.kind == "dir",
                is_file=entry.kind == "file",
            )
        self._ensure_connected()
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
            # Dir listings are part of navigation — drop them so refresh
            # re-lists containers instead of replaying a pinned tree.
            dir_count = len(self._dir_index)
            self._dir_index.clear()
            self._indexed = False
            self._persist_sidecar()
            return len(keys) + dir_count
        if scope == "all":
            count = len(self._index)
            self._index.clear()
            self._dir_index.clear()
            if self._files_root.exists():
                with contextlib.suppress(OSError):
                    shutil.rmtree(self._files_root)
            # Index is gone — caller must re-run ``index()`` / ``prepare``.
            self._indexed = False
            self._ensure_mirror_dirs()
            self._persist_sidecar()
            return count
        raise ValueError(f"unknown scope {scope!r}")

    # ── Internals ───────────────────────────────────────────────────────

    def _pinned_entry(self, key: str) -> _Entry | None:
        """Return a trusted local entry, or None if we must touch remote.

        Pin mode (``ttl_seconds > 0``): any recorded entry wins until an
        **active** refresh (``force_fetch`` / open / user Refresh).
        Strict mode (``ttl_seconds == 0``): never pin.
        Active refresh thread sets ``_tls.force_fetch`` so it always hits remote.
        """
        if getattr(self._tls, "force_fetch", False):
            return None
        if self._ttl_seconds == 0:
            return None
        return self._index.get(key)

    def _pinned_dir(self, key: str) -> _DirEntry | None:
        if getattr(self._tls, "force_fetch", False):
            return None
        if self._ttl_seconds == 0:
            return None
        return self._dir_index.get(key)

    # Back-compat aliases used by older call sites / tests.
    def _fresh_entry(self, key: str) -> _Entry | None:
        return self._pinned_entry(key)

    def _fresh_dir(self, key: str) -> _DirEntry | None:
        return self._pinned_dir(key)

    def _revalidate_file_entry(self, key: str, entry: _Entry) -> bool:
        """Return True when remote file is unchanged; refresh ``fetched_at``.

        Used only in strict mode (``ttl_seconds==0``) when the local mirror
        still has bytes. A single remote ``stat`` is cheaper than re-
        downloading navigation metadata over SSH.
        """
        remote_stat = self._safe_stat(key)
        if remote_stat is None or not remote_stat.is_file:
            return False
        if remote_stat.size != entry.size:
            return False
        # Float mtimes from SSH can differ at sub-second precision across
        # serialisations; treat near-equality as a match.
        if abs(remote_stat.mtime - entry.mtime) > 1e-3:
            return False
        self._record(key, kind="file", size=entry.size, mtime=entry.mtime)
        return True

    def _record(self, key: str, *, kind: str, size: int, mtime: float) -> None:
        with self._lock:
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
        # Pin-until-refresh: load entries as-is.  ``fetched_at`` is advisory
        # only (strict mode / debugging); positive TTL never auto-expires.
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
        """Persist now, or defer to batch exit if inside ``batched()``."""
        if self._defer_persist:
            self._sidecar_dirty = True
            return
        self._write_sidecar()

    @contextlib.contextmanager
    def batched(self) -> Iterator[None]:
        """Defer sidecar writes for the duration of a bulk operation.

        Per-op cache records/invalidations only mark the sidecar dirty; the
        full serialization runs once on exit. Use around bulk walks (e.g.
        :func:`prefetch_workspace_indices`) to avoid O(records²) rewrites.
        Re-entrant: nested ``batched()`` flush only at the outermost exit.
        """
        if self._defer_persist:
            yield  # already batching — inner block is a no-op wrapper
            return
        self._defer_persist = True
        try:
            yield
        finally:
            self._defer_persist = False
            self.flush()

    def flush(self) -> None:
        """Write the sidecar if it has pending (deferred) changes."""
        if self._sidecar_dirty:
            self._write_sidecar()

    def _write_sidecar(self) -> None:
        """Atomically persist ``_index.json``. Missing parent dirs are recreated.

        External ``rm -rf`` of the mirror mid-flight (or a brand-new remote
        open with no prior sidecar) must not leave the cache permanently
        mute — we re-mkdir and retry once before warning.
        """
        payload = {
            "version": _SIDECAR_VERSION,
            "ttl_seconds": self._ttl_seconds,
            "entries": {k: asdict(v) for k, v in self._index.items()},
            "dirs": {
                k: {"names": list(v.names), "fetched_at": v.fetched_at}
                for k, v in self._dir_index.items()
            },
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        tmp = self._sidecar.with_suffix(self._sidecar.suffix + ".tmp")

        def _attempt() -> None:
            self._ensure_mirror_dirs()
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self._sidecar)  # noqa: PTH105

        try:
            _attempt()
            self._sidecar_dirty = False
        except OSError:
            # Parent may have vanished between mkdir and replace (e.g. another
            # process wiped ``~/.molexp/remote_cache/<name>``). Retry once.
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            try:
                _attempt()
                self._sidecar_dirty = False
            except OSError as exc:
                logger.warning("cache sidecar write failed at %s: %s", self._sidecar, exc)
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)


@dataclass
class _PrefetchState:
    warnings: list[PrefetchWarning] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_warning(self, path: str, reason: str) -> None:
        with self.lock:
            self.warnings.append(PrefetchWarning(path=path, reason=reason))


def _prefetch_workers(explicit: int | None) -> int:
    if explicit is not None:
        return max(1, explicit)
    raw = os.environ.get("MOLEXP_PREFETCH_WORKERS", "").strip()
    if raw:
        with contextlib.suppress(ValueError):
            return max(1, int(raw))
    return _DEFAULT_PREFETCH_WORKERS


def _parallel_map[T, R](
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    max_workers: int,
    force_fetch_fs: CachedRemoteFileSystem | None = None,
) -> list[R]:
    """Map *fn* over *items*, parallel when ``max_workers > 1`` and |items| > 1.

    Preserves input order (submit all, collect in order).  When *force_fetch_fs*
    is set, every worker thread inherits ``force_fetch`` so active refreshes
    re-pull remote bytes (``threading.local`` is not inherited otherwise).
    """
    if not items:
        return []
    if max_workers <= 1 or len(items) == 1:
        return [fn(item) for item in items]
    workers = min(max_workers, len(items))

    def _init() -> None:
        if force_fetch_fs is not None:
            force_fetch_fs._tls.force_fetch = True

    with ThreadPoolExecutor(max_workers=workers, initializer=_init) as pool:
        futures = [pool.submit(fn, item) for item in items]
        return [fut.result() for fut in futures]


def prefetch_workspace_indices(
    workspace: Workspace,
    *,
    max_workers: int | None = None,
    on_file: Callable[[str], None] | None = None,
) -> list[PrefetchWarning]:
    """Outside-in parallel walk of entity metadata through ``workspace._fs``.

    Levels (each level fully completes before the next — outer → inner):

    1. **Workspace** — ``workspace.json`` + ``project.json`` index +
       ``listdir(projects/)``.
    2. **Projects** — all ``project.json`` in parallel, then per-project
       experiment indexes + ``listdir(experiments/)`` in parallel.
    3. **Experiments** — all ``experiment.json`` in parallel, then per-
       experiment run indexes + ``listdir(runs/)`` in parallel.
    4. **Runs** — all ``run.json`` in parallel.

    Concurrency is per level (default 8 workers; ``MOLEXP_PREFETCH_WORKERS``
    or *max_workers*).  When the FS is a :class:`CachedRemoteFileSystem`,
    call under :meth:`~CachedRemoteFileSystem.force_fetch` so an **active**
    refresh re-pulls remote bytes instead of replaying the pin.

    *on_file* is invoked once per successfully force-fetched file path
    (progress bar ticks).

    Missing or unreadable nodes become :class:`PrefetchWarning` entries;
    the walk continues so one bad project does not blank the tree.

    Returns:
        Warnings collected during the walk (order not guaranteed under
        parallel execution).
    """
    state = _PrefetchState()
    fs = workspace._fs
    root = str(workspace.root)
    workers = _prefetch_workers(max_workers)
    # Propagate active-refresh force_fetch into worker threads (TLS is not
    # inherited by ThreadPoolExecutor workers).
    force_fs: CachedRemoteFileSystem | None = None
    if isinstance(fs, CachedRemoteFileSystem) and getattr(fs._tls, "force_fetch", False):
        force_fs = fs

    batch = (
        fs.batched()  # ty: ignore[call-non-callable]
        if hasattr(fs, "batched")
        else contextlib.nullcontext()
    )
    # Callers that already entered force_fetch (index/refresh) keep it;
    # bare prefetch still benefits from parallel structure on any FS.
    with batch:
        # ── L0: workspace root (serial — tiny) ──────────────────────────
        _safe_read(fs, fs.join(root, "workspace.json"), state, on_file=on_file)
        projects_dir = fs.join(root, "projects")
        _safe_read(fs, fs.join(root, "project.json"), state, warn_on_missing=False, on_file=on_file)
        try:
            project_names = list(fs.listdir(projects_dir))
        except FileNotFoundError:
            return list(state.warnings)
        except Exception as exc:
            state.add_warning(projects_dir, str(exc))
            return list(state.warnings)

        # ── L1: project.json in parallel ────────────────────────────────
        def _load_project(name: str) -> str | None:
            meta = fs.join(projects_dir, name, "project.json")
            return name if _safe_read(fs, meta, state, on_file=on_file) is not None else None

        healthy_projects = [
            n
            for n in _parallel_map(
                _load_project,
                project_names,
                max_workers=workers,
                force_fetch_fs=force_fs,
            )
            if n
        ]

        # ── L1b: listdir experiments/ per project (parallel) ────────────
        def _list_experiments(project_name: str) -> list[tuple[str, str]]:
            project_dir = fs.join(projects_dir, project_name)
            experiments_dir = fs.join(project_dir, "experiments")
            _safe_read(
                fs,
                fs.join(project_dir, "experiment.json"),
                state,
                warn_on_missing=False,
                on_file=on_file,
            )
            try:
                names = fs.listdir(experiments_dir)
            except FileNotFoundError:
                return []
            except Exception as exc:
                state.add_warning(experiments_dir, str(exc))
                return []
            return [(project_name, n) for n in names]

        exp_pairs: list[tuple[str, str]] = []
        for pairs in _parallel_map(
            _list_experiments,
            healthy_projects,
            max_workers=workers,
            force_fetch_fs=force_fs,
        ):
            exp_pairs.extend(pairs)

        # ── L2: experiment.json in parallel ─────────────────────────────
        def _load_experiment(pair: tuple[str, str]) -> tuple[str, str] | None:
            project_name, exp_name = pair
            meta = fs.join(projects_dir, project_name, "experiments", exp_name, "experiment.json")
            return pair if _safe_read(fs, meta, state, on_file=on_file) is not None else None

        healthy_exps = [
            p
            for p in _parallel_map(
                _load_experiment,
                exp_pairs,
                max_workers=workers,
                force_fetch_fs=force_fs,
            )
            if p
        ]

        # ── L2b: listdir runs/ per experiment (parallel) ────────────────
        def _list_runs(pair: tuple[str, str]) -> list[tuple[str, str, str]]:
            project_name, exp_name = pair
            experiment_dir = fs.join(projects_dir, project_name, "experiments", exp_name)
            runs_dir = fs.join(experiment_dir, "runs")
            _safe_read(
                fs,
                fs.join(experiment_dir, "run.json"),
                state,
                warn_on_missing=False,
                on_file=on_file,
            )
            try:
                names = fs.listdir(runs_dir)
            except FileNotFoundError:
                return []
            except Exception as exc:
                state.add_warning(runs_dir, str(exc))
                return []
            return [(project_name, exp_name, n) for n in names]

        run_triples: list[tuple[str, str, str]] = []
        for triples in _parallel_map(
            _list_runs, healthy_exps, max_workers=workers, force_fetch_fs=force_fs
        ):
            run_triples.extend(triples)

        # ── L3: run.json in parallel (innermost — usually the bulk) ─────
        def _load_run(triple: tuple[str, str, str]) -> None:
            project_name, exp_name, run_name = triple
            meta = fs.join(
                projects_dir,
                project_name,
                "experiments",
                exp_name,
                "runs",
                run_name,
                "run.json",
            )
            _safe_read(fs, meta, state, on_file=on_file)

        _parallel_map(_load_run, run_triples, max_workers=workers, force_fetch_fs=force_fs)

    return list(state.warnings)


def _safe_read(
    fs: FileSystem,
    path: str,
    state: _PrefetchState,
    *,
    warn_on_missing: bool = True,
    on_file: Callable[[str], None] | None = None,
) -> str | None:
    try:
        text = fs.read_text(path)
    except FileNotFoundError as exc:
        if warn_on_missing:
            state.add_warning(path, f"not found: {exc}")
        return None
    except Exception as exc:
        state.add_warning(path, str(exc))
        return None
    if on_file is not None:
        with contextlib.suppress(Exception):
            on_file(path)
    return text


def _read_container_children(
    fs: FileSystem,
    *,
    container_dir: str,
    index_path: str,
    per_child_metadata: str,
    state: _PrefetchState,
    max_workers: int = 1,
) -> list[str]:
    """Warm the children-index, then list the container (optional parallel meta).

    Kept for callers/tests that target a single container.  The main walk
    uses the outside-in levels in :func:`prefetch_workspace_indices`.
    """
    _safe_read(fs, index_path, state, warn_on_missing=False)
    try:
        names = list(fs.listdir(container_dir))
    except FileNotFoundError:
        return []
    except Exception as exc:
        state.add_warning(container_dir, str(exc))
        return []

    def _one(name: str) -> str | None:
        meta_path = fs.join(container_dir, name, per_child_metadata)
        return name if _safe_read(fs, meta_path, state) is not None else None

    return [n for n in _parallel_map(_one, names, max_workers=max_workers) if n]
